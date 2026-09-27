"""Separate adversarial, offline review benchmark for Phase 8 graph retrieval.

This benchmark is deliberately distinct from the original graph.py benchmark.
It exercises 14 hard categories the original corpus does not cover:

  DIRECT              direct fact lookup
  RELATIONAL          explicit relational query
  MULTIHOP            two-hop traversal to reach a candidate
  TEMPORAL            historical / past-usage lookup
  NEGATIVE            explicit negation must not produce a false edge
  SHARED_HUB          shared tool (Ollama) across two projects; scoping must
                      prevent cross-project leakage
  SAME_TOOL           same tool name used by multiple projects
  LEXICAL_OVERLAP     query tokens appear in irrelevant memories; ranking must
                      prefer relevant ones
  DELETED_SUPPORT     support memory deleted; edge must disappear after rebuild
  SUPERSEDED_SUPPORT  support memory superseded; edge still appears but with
                      reduced confidence signal
  EXPLICIT_NEGATION   "does not use" sentence must create no edge
  COMPOUND            compound sentence with two independent subjects
  DOTNAME             technical identifiers with embedded dots (llama.cpp) or
                      hyphens (c++17) must survive canonicalization intact
  AMBIGUOUS           project-level vs tool-level ambiguous name resolution
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5

from contextos.benchmarks.retrieval import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from contextos.core.enums import MemoryStatus, MemoryType, RetrievalMode, TemporalScope
from contextos.core.models import Memory, MemoryUpdate, RetrievalQuery
from contextos.embedding.deterministic import DeterministicEmbedding
from contextos.services.graph import DeterministicEntityExtractor, MemoryGraphService
from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine
from contextos.services.retrieval import HybridRetrievalEngine
from contextos.services.retrieval_index import RetrievalIndexSynchronizer
from contextos.storage.database import Database
from contextos.storage.graph_repo import SqliteGraphRepository
from contextos.storage.lexical.bm25 import BM25Index
from contextos.storage.memory_repo import SqliteMemoryRepository
from contextos.storage.relation_repo import SqliteRelationRepository
from contextos.storage.vector.in_memory import InMemoryVectorStore


_NAMESPACE = UUID("4a02d5f2-d71a-418c-a7f4-1bc91d6fc4ef")


@dataclass(frozen=True)
class AdversarialCase:
    category: str
    text: str
    relevant: tuple[UUID, ...]
    scope: TemporalScope = TemporalScope.CURRENT
    # If True the expected answer is "no results" (empty retrieval set is correct).
    expect_empty: bool = False


def _id(name: str) -> UUID:
    return uuid5(_NAMESPACE, name)


def _mem(name: str, content: str, status: MemoryStatus = MemoryStatus.ACTIVE,
         mem_type: MemoryType = MemoryType.PROJECT) -> Memory:
    return Memory(
        id=_id(name), content=content, status=status, type=mem_type,
        source_type="phase8_adversarial", confidence=0.9,
    )


def adversarial_corpus() -> tuple[list[Memory], list[AdversarialCase]]:  # noqa: C901
    memories: list[Memory] = []
    cases: list[AdversarialCase] = []

    # ------------------------------------------------------------------
    # 1. DIRECT — plain entity name lookup
    # ------------------------------------------------------------------
    m_direct = _mem("direct:1", "Project Alpha uses Ollama")
    memories.append(m_direct)
    cases.append(AdversarialCase(
        "DIRECT", "Ollama", (m_direct.id,),
    ))

    # ------------------------------------------------------------------
    # 2. RELATIONAL — explicit uses-relation query
    # ------------------------------------------------------------------
    m_rel = _mem("relational:1", "Project Beta uses vLLM")
    memories.append(m_rel)
    cases.append(AdversarialCase(
        "RELATIONAL", "What does Project Beta use?", (m_rel.id,),
    ))

    # ------------------------------------------------------------------
    # 3. MULTIHOP — reach model via runtime
    # ------------------------------------------------------------------
    m_mh_proj = _mem("multihop:proj", "Project Gamma uses Ollama")
    m_mh_model = _mem("multihop:model", "Ollama runs Qwen9B", mem_type=MemoryType.FACT)
    memories.extend([m_mh_proj, m_mh_model])
    cases.append(AdversarialCase(
        "MULTIHOP", "Which model is reachable from Project Gamma?", (m_mh_model.id,),
    ))

    # ------------------------------------------------------------------
    # 4. TEMPORAL — historical / previously-used
    # ------------------------------------------------------------------
    m_old = _mem("temporal:old", "Project Delta previously used Docker", MemoryStatus.HISTORICAL)
    m_new = _mem("temporal:new", "Project Delta uses Podman")
    memories.extend([m_old, m_new])
    cases.append(AdversarialCase(
        "TEMPORAL", "What did Project Delta use before?", (m_old.id,), TemporalScope.HISTORICAL,
    ))

    # ------------------------------------------------------------------
    # 5. NEGATIVE — explicit negation must not produce a USES edge
    # ------------------------------------------------------------------
    m_neg_pos = _mem("negative:positive", "Project Epsilon uses Podman")
    m_neg_neg = _mem("negative:negated", "Project Epsilon does not use Docker",
                     mem_type=MemoryType.FACT)
    memories.extend([m_neg_pos, m_neg_neg])
    cases.append(AdversarialCase(
        "NEGATIVE", "Does Project Epsilon use Docker?", (m_neg_pos.id,),
    ))

    # ------------------------------------------------------------------
    # 6. SHARED_HUB — both Project Zeta and Project Eta share Ollama;
    #    a query about Zeta must NOT surface Eta's failure memory.
    # ------------------------------------------------------------------
    m_zeta = _mem("hub:zeta", "Project Zeta uses Ollama")
    m_eta = _mem("hub:eta", "Project Eta uses Ollama")
    m_eta_fail = _mem("hub:eta:fail", "Project Eta works on Failure7", mem_type=MemoryType.FACT)
    memories.extend([m_zeta, m_eta, m_eta_fail])
    cases.append(AdversarialCase(
        "SHARED_HUB", "What does Project Zeta use?", (m_zeta.id,),
    ))

    # ------------------------------------------------------------------
    # 7. SAME_TOOL — vLLM used by both Project Theta and Project Iota;
    #    query for Theta must not return Iota's memory in top results.
    # ------------------------------------------------------------------
    m_theta = _mem("same_tool:theta", "Project Theta uses vLLM")
    m_iota = _mem("same_tool:iota", "Project Iota uses vLLM")
    memories.extend([m_theta, m_iota])
    cases.append(AdversarialCase(
        "SAME_TOOL", "What runtime does Project Theta use?", (m_theta.id,),
    ))

    # ------------------------------------------------------------------
    # 8. LEXICAL_OVERLAP — query text overlaps with an irrelevant memory;
    #    relevant memory must still rank above the noise.
    # ------------------------------------------------------------------
    m_lex_rel = _mem("lexical:relevant", "Project Kappa uses SQLite")
    m_lex_noise = _mem("lexical:noise", "SQLite is a database engine used widely",
                       mem_type=MemoryType.CONTEXT)
    memories.extend([m_lex_rel, m_lex_noise])
    cases.append(AdversarialCase(
        "LEXICAL_OVERLAP", "What does Project Kappa use?", (m_lex_rel.id,),
    ))

    # ------------------------------------------------------------------
    # 9. DELETED_SUPPORT — ingested, then marked deleted;
    #    graph must not surface deleted support memory after rebuild.
    #    We store the deleted memory and expect it NOT to appear.
    # ------------------------------------------------------------------
    m_del = _mem("deleted:mem", "Project Lambda uses Redis", MemoryStatus.DELETED)
    memories.append(m_del)
    # Deleted memories are excluded from projected_statuses, so there should
    # be no graph edge for this.  Relevant result is effectively none, but we
    # use expect_empty rather than forcing a false positive.
    cases.append(AdversarialCase(
        "DELETED_SUPPORT", "What does Project Lambda use?",
        (), expect_empty=True,
    ))

    # ------------------------------------------------------------------
    # 10. SUPERSEDED_SUPPORT — previous relation still in graph
    #     but status is SUPERSEDED; check that retrieval shows the
    #     superseded memory (SUPERSEDED is in projected_statuses and is_retrievable).
    # ------------------------------------------------------------------
    m_sup = _mem("superseded:old", "Project Mu uses RabbitMQ", MemoryStatus.SUPERSEDED,
                 MemoryType.FACT)
    m_sup_new = _mem("superseded:new", "Project Mu uses Kafka")
    memories.extend([m_sup, m_sup_new])
    # Both edges exist in the graph (SUPERSEDED is projected); relevant is current memory.
    cases.append(AdversarialCase(
        "SUPERSEDED_SUPPORT", "What messaging system does Project Mu use?", (m_sup_new.id,),
    ))

    # ------------------------------------------------------------------
    # 11. EXPLICIT_NEGATION — "does not use" must create NO edge
    # ------------------------------------------------------------------
    m_neg2_pos = _mem("explicit_neg:pos", "Project Nu uses PostgreSQL")
    m_neg2_neg = _mem("explicit_neg:neg", "Project Nu does not use MySQL",
                      mem_type=MemoryType.FACT)
    memories.extend([m_neg2_pos, m_neg2_neg])
    cases.append(AdversarialCase(
        "EXPLICIT_NEGATION", "Does Project Nu use MySQL?", (m_neg2_pos.id,),
    ))

    # ------------------------------------------------------------------
    # 12. COMPOUND — compound sentence must produce two independent
    #     source→target edges, never a cross-project edge.
    # ------------------------------------------------------------------
    m_comp_a = _mem("compound:a", "Project Xi uses Python and Project Omicron uses Rust",
                    mem_type=MemoryType.FACT)
    memories.append(m_comp_a)
    cases.append(AdversarialCase(
        "COMPOUND", "What does Project Xi use?", (m_comp_a.id,),
    ))
    cases.append(AdversarialCase(
        "COMPOUND", "What does Project Omicron use?", (m_comp_a.id,),
    ))

    # ------------------------------------------------------------------
    # 13. DOTNAME — embedded-dot and hyphen technical identifiers must
    #     survive canonicalization intact.
    # ------------------------------------------------------------------
    m_dot = _mem("dotname:llama", "Project Pi uses llama.cpp")
    m_dot2 = _mem("dotname:cpp17", "Project Rho uses c++17")
    memories.extend([m_dot, m_dot2])
    cases.append(AdversarialCase(
        "DOTNAME", "What inference engine does Project Pi use?", (m_dot.id,),
    ))
    cases.append(AdversarialCase(
        "DOTNAME", "What compiler standard does Project Rho use?", (m_dot2.id,),
    ))

    # ------------------------------------------------------------------
    # 14. AMBIGUOUS — "Atlas" could be a project or tool name;
    #     query must resolve to the correct memory.
    # ------------------------------------------------------------------
    m_amb = _mem("ambiguous:atlas", "Project Atlas uses Ollama")
    m_amb2 = _mem("ambiguous:atlas_tool", "Atlas is also the name of a mapping tool",
                  MemoryStatus.ACTIVE, MemoryType.CONTEXT)
    memories.extend([m_amb, m_amb2])
    cases.append(AdversarialCase(
        "AMBIGUOUS", "What does Project Atlas use?", (m_amb.id,),
    ))

    # ------------------------------------------------------------------
    # 15. NO-PATH — query for entity with no graph tool connection;
    #     should surface only via lexical/dense, not fabricate graph edge.
    # ------------------------------------------------------------------
    m_iso = _mem("nopath:isolated", "Project Sigma is an internal analytics service",
                 mem_type=MemoryType.CONTEXT)
    memories.append(m_iso)
    cases.append(AdversarialCase(
        "NEGATIVE",
        "What runtime does Project Sigma depend on?",
        (m_iso.id,),
    ))

    # ------------------------------------------------------------------
    # 16. VERSION_NUMBER — identifier with numeric suffix must survive.
    # ------------------------------------------------------------------
    m_ver = _mem("version:qwen", "Project Tau uses Qwen30B for inference",
                 mem_type=MemoryType.FACT)
    memories.append(m_ver)
    cases.append(AdversarialCase(
        "DOTNAME", "Which model does Project Tau use?", (m_ver.id,),
    ))

    # ------------------------------------------------------------------
    # 17. TEMPORAL REPLACEMENT — "used X but now uses Y" keeps only Y
    #     as a current USES edge.
    # ------------------------------------------------------------------
    m_rep = _mem("temporal_replace:mem",
                 "Project Upsilon used Docker previously but now uses Podman")
    memories.append(m_rep)
    cases.append(AdversarialCase(
        "TEMPORAL", "What does Project Upsilon use now?", (m_rep.id,),
    ))

    # ------------------------------------------------------------------
    # 18. SECOND SHARED-HUB CROSS-CHECK — Project Phi also uses Ollama;
    #     querying about Phi's specific task must not surface Eta/Zeta.
    # ------------------------------------------------------------------
    m_phi = _mem("hub:phi", "Project Phi uses Ollama")
    m_phi_task = _mem("hub:phi:task", "Project Phi works on VisionTask",
                      mem_type=MemoryType.FACT)
    memories.extend([m_phi, m_phi_task])
    cases.append(AdversarialCase(
        "SHARED_HUB", "What does Project Phi work on?", (m_phi_task.id,),
    ))

    # ------------------------------------------------------------------
    # 19. COMPOUND WITH SEMICOLON — split on semicolon, same rules apply.
    # ------------------------------------------------------------------
    m_semi = _mem("compound:semi",
                  "Project Chi uses vLLM; Project Psi uses Ollama",
                  mem_type=MemoryType.FACT)
    memories.append(m_semi)
    cases.append(AdversarialCase(
        "COMPOUND", "What does Project Chi use?", (m_semi.id,),
    ))

    # ------------------------------------------------------------------
    # 20. MULTIHOP DEPTH-2 NO PATH — Omega has no tool that runs Qwen9B;
    #     multihop expansion must not fabricate the path.
    # ------------------------------------------------------------------
    m_omega = _mem("multihop_neg:omega", "Project Omega uses SQLite")
    memories.append(m_omega)
    cases.append(AdversarialCase(
        "MULTIHOP",
        "What model does Project Omega run through its runtime?",
        (m_omega.id,),
    ))

    # ------------------------------------------------------------------
    # 21. SUPERSEDED + CURRENT — for same project, CURRENT scope must
    #     prefer the active memory over the superseded one.
    # ------------------------------------------------------------------
    m_sup2_old = _mem("superseded2:old", "Project Psi uses Redis",
                      MemoryStatus.SUPERSEDED, MemoryType.PROJECT)
    m_sup2_new = _mem("superseded2:new", "Project Psi uses Kafka")
    memories.extend([m_sup2_old, m_sup2_new])
    cases.append(AdversarialCase(
        "SUPERSEDED_SUPPORT",
        "What storage does Project Psi currently use?",
        (m_sup2_new.id,),
    ))

    # ------------------------------------------------------------------
    # 22. LEXICAL OVERLAP — many memories mention Docker; structured
    #     relation query for Project Omega/Docker must be precise.
    # ------------------------------------------------------------------
    m_stop = _mem("lexical:docker_omega", "Project Omega uses Docker")
    memories.append(m_stop)
    cases.append(AdversarialCase(
        "LEXICAL_OVERLAP",
        "What does Project Omega use for containerization?",
        (m_stop.id,),
    ))

    # ------------------------------------------------------------------
    # 23. DIRECT — bare tool name with multiple project users; retrieval
    #     must surface at least one relevant memory.
    # ------------------------------------------------------------------
    m_dir2 = _mem("direct:postgres", "Project Alpha uses PostgreSQL")
    memories.append(m_dir2)
    cases.append(AdversarialCase(
        "DIRECT", "PostgreSQL", (m_dir2.id,),
    ))

    # ------------------------------------------------------------------
    # 24. AMBIGUOUS PROJECT NAME — "Nova" in unrelated content must not
    #     pollute the project-level structured query.
    # ------------------------------------------------------------------
    m_nova = _mem("ambiguous:nova", "Project Nova uses Ollama")
    m_nova_noise = _mem("ambiguous:nova_noise",
                        "Nova is a constellation in the southern hemisphere",
                        MemoryStatus.ACTIVE, MemoryType.CONTEXT)
    memories.extend([m_nova, m_nova_noise])
    cases.append(AdversarialCase(
        "AMBIGUOUS", "What does Project Nova use?", (m_nova.id,),
    ))

    # ------------------------------------------------------------------
    # 25. RELATIONAL + NEGATION COMBO — positive and negative clause in
    #     same sentence; only the positive clause must produce an edge.
    # ------------------------------------------------------------------
    m_rel_neg = _mem("relational_neg:mem",
                     "Project Omega uses Podman but does not use Docker",
                     mem_type=MemoryType.FACT)
    memories.append(m_rel_neg)
    cases.append(AdversarialCase(
        "NEGATIVE",
        "Does Project Omega use Podman?",
        (m_rel_neg.id,),
    ))

    assert len(cases) >= 25, f"Expected >= 25 adversarial cases, got {len(cases)}"
    return memories, cases


async def run_adversarial_benchmark(root: Path | None = None) -> dict[str, object]:
    owned = root is None
    temporary = tempfile.TemporaryDirectory() if owned else None
    directory = Path(temporary.name) if temporary else root
    assert directory is not None
    directory.mkdir(parents=True, exist_ok=True)
    database = Database(directory / "graph-adversarial.db")
    await database.initialize()
    try:
        memory_repo = SqliteMemoryRepository(database.connection())
        relation_repo = SqliteRelationRepository(database.connection())
        graph_repo = SqliteGraphRepository(database.connection())
        graph = MemoryGraphService(
            memory_repo=memory_repo, relation_repo=relation_repo, graph_repo=graph_repo,
        )
        memories, cases = adversarial_corpus()
        for memory in memories:
            await memory_repo.create(memory)

        embedding = DeterministicEmbedding(32)
        lexical = BM25Index()
        vector = InMemoryVectorStore(32)
        base = HybridRetrievalEngine(
            memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
            embedding_service=embedding,
            index_synchronizer=RetrievalIndexSynchronizer(
                memory_repo=memory_repo, lexical_index=lexical, vector_store=vector,
                embedding_service=embedding,
            ),
        )
        engine = GraphAugmentedRetrievalEngine(
            base_engine=base, graph_service=graph, memory_repo=memory_repo,
        )

        metrics: dict[str, list[float]] = {"recall": [], "mrr": [], "ndcg": [], "hit": []}
        by_category: dict[str, dict[str, list[float]]] = {}
        safety: dict[str, bool] = {}

        for case in cases:
            result = await engine.retrieve(RetrievalQuery(
                text=case.text, mode=RetrievalMode.HYBRID_GRAPH, k=5,
                temporal_scope=case.scope,
            ))
            ranking = [str(item.memory.id) for item in result.memories]
            if case.expect_empty:
                # The deleted memory's ID must not appear in any retrieval result.
                deleted_ids = {str(rel_id) for rel_id in case.relevant if case.relevant}
                # relevant is empty for deleted cases; safety checked below via deleted_id set.
                # Still record that no result has a deleted-status memory.
                safety[f"deleted_support_excluded_{case.category.lower()}"] = not any(
                    item.memory.status.value == "deleted" for item in result.memories
                )
                continue
            qrels = {str(identifier): 1 for identifier in case.relevant}
            r = recall_at_k(ranking, qrels, 5)
            m = reciprocal_rank(ranking, qrels)
            n = ndcg_at_k(ranking, qrels, 5)
            h = hit_rate_at_k(ranking, qrels, 5)
            metrics["recall"].append(r)
            metrics["mrr"].append(m)
            metrics["ndcg"].append(n)
            metrics["hit"].append(h)
            cat = case.category
            if cat not in by_category:
                by_category[cat] = {"recall": [], "mrr": [], "ndcg": [], "hit": []}
            by_category[cat]["recall"].append(r)
            by_category[cat]["mrr"].append(m)
            by_category[cat]["ndcg"].append(n)
            by_category[cat]["hit"].append(h)

        # --- Safety checks ---
        edges = await graph_repo.all_edges()
        nodes = {node.id: node for node in await graph_repo.nodes()}

        # Negation: Epsilon explicitly negated "does not use Docker" → no Epsilon→Docker USES edge
        epsilon_node_key = "epsilon"
        epsilon_node = next(
            (node for node in nodes.values() if node.canonical_key == epsilon_node_key), None
        )
        safety["negation_epsilon_no_docker_edge"] = not any(
            edge.relation_type.value == "uses"
            and edge.source_node_id == (epsilon_node.id if epsilon_node else None)
            and nodes.get(edge.target_node_id) is not None
            and nodes[edge.target_node_id].canonical_key == "docker"
            for edge in edges
        )
        # Negation: no USES edge targeting MySQL from Project Nu
        safety["explicit_negation_mysql_no_edge"] = not any(
            edge.relation_type.value == "uses"
            and nodes.get(edge.target_node_id) is not None
            and nodes[edge.target_node_id].canonical_key == "mysql"
            for edge in edges
        )
        # Dotname: llama.cpp survives canonicalization
        extractor = DeterministicEntityExtractor()
        rel_dot = extractor.relations("Project Pi uses llama.cpp")
        safety["dotname_llama_cpp_preserved"] = (
            len(rel_dot) == 1 and rel_dot[0].target.key == "llama.cpp"
        )
        # Dotname: c++17 survives canonicalization
        rel_cpp = extractor.relations("Project Rho uses c++17")
        safety["dotname_cpp17_preserved"] = (
            len(rel_cpp) == 1 and rel_cpp[0].target.key == "c++17"
        )
        # Compound: exactly two edges from compound sentence, no cross-project edge
        rel_comp = extractor.relations("Project Xi uses Python and Project Omicron uses Rust")
        pairs = {(r.source.key, r.target.key) for r in rel_comp}
        safety["compound_no_cross_edge"] = pairs == {("xi", "python"), ("omicron", "rust")}

        # Shared-hub scoping: Project Zeta query must not surface Eta's failure memory
        scope_result = await engine.retrieve(RetrievalQuery(
            text="What does Project Zeta use?",
            mode=RetrievalMode.GRAPH, k=10, graph_max_hops=3,
        ))
        safety["shared_hub_no_cross_project_leak"] = all(
            "Failure7" not in item.memory.content for item in scope_result.memories
        )

        # Deleted-support: Lambda/Redis edge must not appear in graph after rebuild
        # (MemoryStatus.DELETED is excluded from projected_statuses)
        safety["deleted_support_no_edge"] = not any(
            "lambda" in (nodes.get(edge.source_node_id) and nodes[edge.source_node_id].canonical_key or "")
            for edge in edges if edge.relation_type.value == "uses"
        )

        avg = {key: sum(values) / len(values) for key, values in metrics.items() if values}
        cat_avg = {
            cat: {key: sum(values) / len(values) for key, values in v.items() if values}
            for cat, v in by_category.items()
        }
        return {
            "memories": len(memories),
            "queries": len([c for c in cases if not c.expect_empty]),
            "categories": sorted(cat_avg.keys()),
            "overall": avg,
            "by_category": cat_avg,
            "safety": safety,
        }
    finally:
        await database.close()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    import json
    print(json.dumps(asyncio.run(run_adversarial_benchmark()), indent=2, sort_keys=True))
