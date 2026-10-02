"""Deterministic property-graph projection and bounded traversal."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid5

from contextos.core.enums import (
    GraphNodeType,
    GraphRelationType,
    MemoryStatus,
)
from contextos.core.models import (
    GraphCandidateEvidence,
    GraphEdge,
    GraphEdgeSupport,
    GraphExpansion,
    GraphNode,
    GraphPath,
    GraphPathEdge,
    GraphPathNode,
    Memory,
    MemoryFilters,
)
from contextos.core.protocols import MemoryRepository, RelationRepository
from contextos.storage.graph_repo import SqliteGraphRepository


GRAPH_NAMESPACE = UUID("1e842c72-7e40-4f2a-b905-2f7e9bc10325")
_TOKEN = r"[A-Za-z][A-Za-z0-9+#._-]*"
_ENTITY = rf"(?:Project\s+)?{_TOKEN}(?:\s+{_TOKEN}){{0,2}}"
_NEGATED = re.compile(r"\b(?:does\s+not|do\s+not|did\s+not|never|no\s+longer|not)\b", re.I)
_RELATION = re.compile(
    rf"\b(?P<source>{_ENTITY})\s+"
    r"(?:(?:currently|now|previously|formerly)\s+)?"
    r"(?P<verb>uses?|used|runs?|depends\s+on|is\s+part\s+of|belongs\s+to|works\s+on)\s+"
    r"(?P<target>[^,;!?]+?)(?=[,;!?]|$)",
    re.I,
)
_CURRENT_REPLACEMENT = re.compile(
    rf"\b(?P<source>(?:Project\s+)?{_TOKEN})\s+used\s+.+?\s+previously\s+"
    rf"but\s+now\s+uses\s+(?P<target>{_TOKEN}(?:\s+[A-Za-z0-9][A-Za-z0-9+#._-]*)?)",
    re.I,
)
_KNOWN_TOOLS = {
    "ollama", "llama.cpp", "vllm", "docker", "podman", "python", "rust",
    "c++17", "qwen9b", "qwen30b", "qwen", "postgresql", "sqlite",
}
_STOP_LABELS = {
    "user", "the", "a", "an", "current", "previous", "larger", "model",
    "runtime", "tool", "memory", "project", "failed", "failure", "concise",
    "which", "what", "when", "where", "why", "how",
}
_RELATION_MAP = {
    "use": GraphRelationType.USES,
    "uses": GraphRelationType.USES,
    "used": GraphRelationType.USES,
    "run": GraphRelationType.RUNS,
    "runs": GraphRelationType.RUNS,
    "depends on": GraphRelationType.DEPENDS_ON,
    "is part of": GraphRelationType.PART_OF,
    "belongs to": GraphRelationType.BELONGS_TO,
    "works on": GraphRelationType.WORKS_ON,
}


def canonical_key(label: str) -> str:
    value = re.sub(r"^project\s+", "", label.strip(), flags=re.I)
    value = " ".join(value.casefold().split())
    return re.sub(r"\b(qwen)\s+(\d+b)\b", r"\1\2", value)


def stable_node_id(node_type: GraphNodeType, key: str) -> UUID:
    return uuid5(GRAPH_NAMESPACE, f"node:{node_type.value}:{key}")


def stable_edge_id(
    source: UUID, target: UUID, relation: GraphRelationType, scope_key: str | None,
) -> UUID:
    return uuid5(GRAPH_NAMESPACE, f"edge:{source}:{target}:{relation.value}:{scope_key or ''}")


def _safe_node_label(label: str | None) -> str | None:
    if not label or label.startswith("memory:"):
        return None
    clean = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\)|.)", "", label)
    clean = "".join(c for c in clean if c.isprintable() and not (0x202A <= ord(c) <= 0x2069))[:120].strip()
    if re.search(
        r"(?i)(?:[a-z]:[\\/]|\\\\|(?:^|\s)/(?:[^/\s]+/)+|"
        r"[a-z][a-z0-9+.-]*://|\b(?:api[_ -]?key|password|passwd|secret|token|authorization)\s*[:=]|"
        r"\bbearer\s+\S+)",
        clean,
    ):
        return None
    return clean or None



@dataclass(frozen=True)
class ExtractedEntity:
    label: str
    key: str
    node_type: GraphNodeType


@dataclass(frozen=True)
class ExtractedRelation:
    source: ExtractedEntity
    target: ExtractedEntity
    relation_type: GraphRelationType
    scope_key: str | None
    confidence: float


class DeterministicEntityExtractor:
    """Extract only named entities and explicitly worded relations."""

    def entities(self, text: str) -> list[ExtractedEntity]:
        found: dict[tuple[GraphNodeType, str], ExtractedEntity] = {}
        for relation in self.relations(text):
            for entity in (relation.source, relation.target):
                found[(entity.node_type, entity.key)] = entity
        for match in re.finditer(r"\bProject\s+([A-Za-z][A-Za-z0-9_-]*)", text, re.I):
            entity = self._entity(match.group(1), GraphNodeType.PROJECT)
            found[(entity.node_type, entity.key)] = entity
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9+#._-]*", text):
            label = match.group(0)
            key = canonical_key(label)
            if key in _KNOWN_TOOLS:
                entity = self._entity(label, GraphNodeType.TOOL)
                found[(entity.node_type, entity.key)] = entity
        return sorted(found.values(), key=lambda item: (item.node_type.value, item.key))

    def query_entities(self, text: str) -> list[ExtractedEntity]:
        values = self.entities(text)
        seen = {(item.node_type, item.key) for item in values}
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9+#._-]+\b", text):
            label = match.group(0)
            key = canonical_key(label)
            if key not in _STOP_LABELS:
                node_type = GraphNodeType.TOOL if key in _KNOWN_TOOLS else GraphNodeType.PROJECT
                item = self._entity(label, node_type)
                if (item.node_type, item.key) not in seen:
                    values.append(item)
                    seen.add((item.node_type, item.key))
        return values

    def relations(self, text: str, *, negated: bool = False) -> list[ExtractedRelation]:
        if negated or _NEGATED.search(text):
            return []
        output: list[ExtractedRelation] = []
        # A conjunction beginning another explicit subject is two independent
        # claims, never one long object phrase.  Keep this narrow so ordinary
        # tool names and prose cannot manufacture extra relations.
        clauses = re.split(
            rf"\s+(?:and|;)+\s+(?=(?:Project\s+)?{_TOKEN}\s+(?:uses?|used|runs?|depends\s+on|is\s+part\s+of|belongs\s+to|works\s+on)\b)",
            text,
            flags=re.I,
        )
        if len(clauses) > 1:
            for clause in clauses:
                output.extend(self.relations(clause, negated=negated))
            return output
        replacement = _CURRENT_REPLACEMENT.search(text)
        if replacement:
            source = self._entity(replacement.group("source"), GraphNodeType.PROJECT)
            target = self._entity(replacement.group("target"), GraphNodeType.TOOL)
            return [ExtractedRelation(
                source=source, target=target, relation_type=GraphRelationType.USES,
                scope_key=source.key, confidence=0.95,
            )]
        for match in _RELATION.finditer(text):
            source_label = match.group("source").strip()
            source_label = re.sub(
                r"\s+(?:currently|now|previously|formerly)$", "", source_label,
                flags=re.I,
            )
            target_label = match.group("target").strip().rstrip(".,;:!?")
            target_label = re.split(
                r"\s+(?:instead\s+of|previously|formerly|because|due\s+to|but\s+now)\b",
                target_label, maxsplit=1, flags=re.I,
            )[0]
            target_label = re.sub(
                r"\s+(?:for\s+local\s+inference|locally|on\s+the\s+local\s+machine)$",
                "", target_label, flags=re.I,
            ).strip()
            if not target_label:
                continue
            verb = " ".join(match.group("verb").casefold().split())
            relation_type = _RELATION_MAP.get(verb)
            if relation_type is None:
                continue
            source_key = canonical_key(source_label)
            target_key = canonical_key(target_label)
            source_type = (
                GraphNodeType.TOOL
                if relation_type == GraphRelationType.RUNS or source_key in _KNOWN_TOOLS
                else GraphNodeType.PROJECT
            )
            target_type = (
                GraphNodeType.PROJECT
                if target_label.casefold().startswith("project ")
                else GraphNodeType.TOOL
            )
            source = self._entity(source_label, source_type)
            target = self._entity(target_label, target_type)
            scope = source.key if source.node_type == GraphNodeType.PROJECT else None
            output.append(ExtractedRelation(
                source=source, target=target, relation_type=relation_type,
                scope_key=scope, confidence=0.95,
            ))
        return output

    @staticmethod
    def _entity(label: str, node_type: GraphNodeType) -> ExtractedEntity:
        clean = re.sub(r"^project\s+", "", label.strip(), flags=re.I).rstrip(".,;:!?")
        return ExtractedEntity(label=clean, key=canonical_key(clean), node_type=node_type)


class MemoryGraphService:
    """Build and query a conservative graph derived from authoritative memories."""

    _projected_statuses = (
        MemoryStatus.ACTIVE, MemoryStatus.HISTORICAL, MemoryStatus.SUPERSEDED,
        MemoryStatus.CONTRADICTED, MemoryStatus.EXPIRED,
    )

    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        relation_repo: RelationRepository,
        graph_repo: SqliteGraphRepository,
        extractor: DeterministicEntityExtractor | None = None,
    ) -> None:
        self._memory_repo = memory_repo
        self._relation_repo = relation_repo
        self._graph_repo = graph_repo
        self._extractor = extractor or DeterministicEntityExtractor()
        self._lock = asyncio.Lock()

    async def ensure_current(self, *, force: bool = False) -> bool:
        async with self._lock:
            if not force and not await self._graph_repo.source_is_dirty():
                return False
            memories = await self._memories()
            await self._rebuild_from(memories)
            return True

    async def rebuild(self) -> tuple[int, int, int]:
        await self.ensure_current(force=True)
        return await self._graph_repo.counts()

    async def upsert_memory(self, memory: Memory) -> tuple[int, int, int]:
        del memory
        return await self.rebuild()

    async def remove_memory(self, memory_id: UUID) -> tuple[int, int, int]:
        del memory_id
        return await self.rebuild()

    async def find_entities(self, text: str) -> list[GraphNode]:
        await self.ensure_current()
        keys = {item.key for item in self._extractor.query_entities(text)}
        return await self._graph_repo.find_nodes(keys)

    async def neighbors(
        self,
        node_id: UUID,
        *,
        relation_types: set[GraphRelationType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[GraphEdge]:
        await self.ensure_current()
        edges = await self._graph_repo.edges_for_nodes({node_id})
        return [
            edge for edge in edges
            if edge.confidence >= min_confidence
            and (relation_types is None or edge.relation_type in relation_types)
        ]

    async def expand(
        self,
        *,
        query_text: str,
        seed_memory_ids: list[UUID] | None = None,
        max_hops: int = 2,
        min_confidence: float = 0.6,
        max_nodes: int = 100,
        max_edges: int = 250,
        relation_types: set[GraphRelationType] | None = None,
    ) -> GraphExpansion:
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be between 1 and 3")
        await self.ensure_current()
        query_nodes = await self.find_entities(query_text)
        seed_ids = {node.id for node in query_nodes}
        for memory_id in seed_memory_ids or []:
            seed_ids.add(stable_node_id(GraphNodeType.MEMORY, str(memory_id)))
        if not seed_ids:
            return GraphExpansion()

        all_nodes: dict[UUID, GraphNode] = {}
        for seed in seed_ids:
            node = await self._graph_repo.get_node(seed)
            if node is not None:
                all_nodes[seed] = node

        candidate_scores: dict[UUID, float] = {}
        candidate_paths: dict[UUID, list[GraphPath]] = defaultdict(list)
        visited: set[UUID] = set(seed_ids)
        traversed: set[UUID] = set()
        queue = deque(
            (seed, seed, [seed], [], 1.0, self._seed_scope(all_nodes.get(seed)))
            for seed in sorted(seed_ids, key=str)
        )
        while queue and len(visited) <= max_nodes and len(traversed) < max_edges:
            seed, current, node_path, edge_path, strength, scope = queue.popleft()
            hop = len(edge_path)
            if hop >= max_hops:
                continue
            incident = await self._graph_repo.edges_for_nodes({current})
            adjacent: list[tuple[GraphEdge, UUID]] = []
            for edge in incident:
                if edge.confidence < min_confidence:
                    continue
                if relation_types is not None and edge.relation_type not in relation_types:
                    continue
                if edge.source_node_id == current:
                    adjacent.append((edge, edge.target_node_id))
                elif not edge.directed or edge.relation_type in {
                    GraphRelationType.ABOUT, GraphRelationType.MENTIONS,
                }:
                    adjacent.append((edge, edge.source_node_id))
            degree = max(1, len(adjacent))
            for edge, neighbor in adjacent:
                if len(traversed) >= max_edges:
                    break
                next_scope = scope
                if edge.scope_key:
                    if scope and edge.scope_key != scope:
                        continue
                    next_scope = scope or edge.scope_key
                next_hop = hop + 1
                attenuation = edge.confidence * (0.72 ** next_hop) / (degree ** 0.5)
                contribution = min(1.0, strength * attenuation)
                new_nodes = [*node_path, neighbor]
                new_edges = [*edge_path, edge]
                traversed.add(edge.id)
                if neighbor not in all_nodes:
                    neighbor_node = await self._graph_repo.get_node(neighbor)
                    if neighbor_node is not None:
                        all_nodes[neighbor] = neighbor_node
                support_ids = sorted({item.memory_id for item in edge.supports}, key=str)
                path_nodes: list[GraphPathNode] = []
                for n_id in new_nodes:
                    n_obj = all_nodes.get(n_id)
                    if n_obj is not None:
                        label = _safe_node_label(n_obj.label)
                        project_scope = (
                            _safe_node_label(n_obj.canonical_key)
                            if n_obj.node_type == GraphNodeType.PROJECT else None
                        )
                        path_nodes.append(GraphPathNode(
                            node_id=n_obj.id,
                            node_type=n_obj.node_type,
                            label=label,
                            project_scope=project_scope,
                        ))
                    else:
                        path_nodes.append(GraphPathNode(
                            node_id=n_id,
                            node_type=GraphNodeType.MEMORY,
                            label=None,
                            project_scope=None,
                        ))

                path_edges: list[GraphPathEdge] = []
                for e_obj in new_edges:
                    path_edges.append(GraphPathEdge(
                        edge_type=e_obj.relation_type,
                        confidence=e_obj.confidence,
                        supporting_memory_ids=sorted({s.memory_id for s in e_obj.supports}, key=str),
                        project_scope=_safe_node_label(e_obj.scope_key),
                    ))

                scope_participated = any(e.project_scope is not None for e in path_edges) or any(
                    n.project_scope is not None for n in path_nodes
                )
                scope_match = None
                if scope_participated and scope is not None:
                    scope_match = any(e.project_scope == scope for e in path_edges) or any(
                        n.project_scope == scope for n in path_nodes
                    )

                for memory_id in support_ids:
                    candidate_scores[memory_id] = max(candidate_scores.get(memory_id, 0.0), contribution)
                    candidate_paths[memory_id].append(GraphPath(
                        seed_node_ids=[seed],
                        node_ids=new_nodes,
                        node_types=[all_nodes[node].node_type for node in new_nodes if node in all_nodes],
                        edge_ids=[item.id for item in new_edges],
                        edge_types=[item.relation_type for item in new_edges],
                        hop_count=next_hop,
                        graph_contribution=contribution,
                        source_memory_ids=support_ids,
                        path_nodes=path_nodes,
                        path_edges=path_edges,
                        scope_match=scope_match,
                    ))
                if neighbor not in visited and len(visited) < max_nodes:
                    visited.add(neighbor)
                    queue.append((seed, neighbor, new_nodes, new_edges, contribution, next_scope))

        return GraphExpansion(
            seed_node_ids=sorted(seed_ids, key=str), candidate_scores=candidate_scores,
            candidate_paths={key: value[:3] for key, value in candidate_paths.items()},
            visited_node_ids=sorted(visited, key=str),
            traversed_edge_ids=sorted(traversed, key=str),
        )

    async def _memories(self) -> list[Memory]:
        values: list[Memory] = []
        for status in self._projected_statuses:
            offset = 0
            while True:
                page = await self._memory_repo.list(MemoryFilters(status=status, limit=500, offset=offset))
                values.extend(page)
                if len(page) < 500:
                    break
                offset += len(page)
        return values

    async def _rebuild_from(self, memories: list[Memory]) -> None:
        now = datetime.now(timezone.utc)
        nodes: dict[UUID, GraphNode] = {}
        edge_data: dict[tuple[UUID, UUID, GraphRelationType, str], GraphEdge] = {}

        def add_node(node_type: GraphNodeType, key: str, label: str, metadata: dict | None = None) -> UUID:
            identifier = stable_node_id(node_type, key)
            nodes.setdefault(identifier, GraphNode(
                id=identifier, node_type=node_type, canonical_key=key, label=label,
                metadata=metadata or {}, created_at=now, updated_at=now,
            ))
            return identifier

        def add_edge(
            source: UUID, target: UUID, relation: GraphRelationType, memory: Memory,
            confidence: float, scope: str | None = None,
        ) -> None:
            key = (source, target, relation, scope or "")
            support = GraphEdgeSupport(
                edge_id=stable_edge_id(source, target, relation, scope), memory_id=memory.id,
                confidence=confidence, provenance_event_id=memory.provenance_event_id,
                created_at=now,
            )
            if key in edge_data:
                existing = edge_data[key]
                if all(item.memory_id != memory.id for item in existing.supports):
                    existing.supports.append(support)
                existing.confidence = max(existing.confidence, confidence)
                return
            edge_data[key] = GraphEdge(
                id=support.edge_id, source_node_id=source, target_node_id=target,
                relation_type=relation, confidence=confidence,
                directed=relation not in {
                    GraphRelationType.MENTIONS,
                    GraphRelationType.ABOUT,
                    GraphRelationType.COEXISTS_WITH,
                    GraphRelationType.CONTRADICTS,
                },
                scope_key=scope, supports=[support], created_at=now, updated_at=now,
            )

        by_id = {memory.id: memory for memory in memories}
        for memory in memories:
            entities = self._extractor.entities(memory.content)
            relations = self._extractor.relations(memory.content, negated=memory.negated)
            if not entities:
                continue
            memory_node = add_node(
                GraphNodeType.MEMORY, str(memory.id), f"memory:{memory.id}",
                {"status": memory.status.value},
            )
            primary_keys = {relation.source.key for relation in relations}
            for entity in entities:
                entity_node = add_node(entity.node_type, entity.key, entity.label)
                relation = (
                    GraphRelationType.ABOUT if entity.key in primary_keys else GraphRelationType.MENTIONS
                )
                add_edge(memory_node, entity_node, relation, memory, 0.9)
            for relation in relations:
                source = add_node(relation.source.node_type, relation.source.key, relation.source.label)
                target = add_node(relation.target.node_type, relation.target.key, relation.target.label)
                add_edge(
                    source, target, relation.relation_type, memory,
                    relation.confidence, relation.scope_key,
                )

        seen_relations: set[UUID] = set()
        mapping = {item.value: GraphRelationType(item.value) for item in GraphRelationType}
        for memory in memories:
            for relation in await self._relation_repo.get_relations(memory.id, "outgoing"):
                if relation.id in seen_relations or relation.target_memory_id not in by_id:
                    continue
                seen_relations.add(relation.id)
                graph_type = mapping.get(relation.relation_type.value)
                if graph_type is None:
                    continue
                source = add_node(GraphNodeType.MEMORY, str(relation.source_memory_id), f"memory:{relation.source_memory_id}")
                target = add_node(GraphNodeType.MEMORY, str(relation.target_memory_id), f"memory:{relation.target_memory_id}")
                add_edge(source, target, graph_type, memory, relation.confidence)

        await self._graph_repo.replace_all(list(nodes.values()), list(edge_data.values()))

    @staticmethod
    def _seed_scope(node: GraphNode | None) -> str | None:
        return node.canonical_key if node and node.node_type == GraphNodeType.PROJECT else None
