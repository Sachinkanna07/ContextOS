# Architecture

ContextOS is a single-machine daemon, not a chatbot or hosted account service. The CLI talks to the FastAPI daemon on the configured loopback address. `wire_services()` composes SQLite repositories, privacy/extraction/temporal services, retrieval indexes, graph projection, optimizer, compiler, model routing, telemetry, connectors, explainability, and inspection.

## Write and read paths

```text
CLI / API / opt-in MCP / configured connector
  -> IngestionPipeline (privacy scan, rule-based candidate extraction, provenance event)
  -> TemporalMemoryService.accept() (per-candidate durable transaction)
  -> SQLite memories and relations
  -> retrieval/graph projection invalidation

query -> index synchronization -> BM25 + local dense embeddings -> eligibility
      -> rank fusion -> optional graph augmentation -> token-aware selection
      -> fact compiler -> optional provider routing -> invocation telemetry
```

The rule-based extractor and deterministic graph entity/relation extractor are deliberately narrower than an LLM. Ambiguous or unsupported facts may not be extracted. Multiple candidate writes are individually atomic, not one transaction for the entire request. Connector cursor advancement follows its own failure semantics; see [connectors](connectors.md).

## Storage, retrieval, and time

SQLite schema version 7 stores memories, events, temporal relations, graph projection/supports, connector state, and model-invocation telemetry. The database uses WAL. BM25 and in-memory dense indexes are rebuildable from SQLite. The retrieval index synchronizer checks a persisted-memory fingerprint, then rebuilds as needed. Candidate hydration uses bounded batches on the SQLite repository, with the repository-protocol fallback still supported. This avoids per-candidate SQLite round trips but index synchronization still scales with corpus size.

The current temporal scope excludes stale states by default. Explicit historical/all scopes can retrieve them. The temporal peer lookup checks the latest active same-subject/property peer, without the old 500-row truncation. The graph is a persisted deterministic projection of memories and relations with a dirty bit and bounded traversal (maximum three hops, explicit node/edge limits). Graph-assisted ranking is opt-in, not a global default.

## Context and evidence

The optimizer chooses candidate memories under a budget; the compiler emits bounded facts/context. These are distinct operations: selected memories need not contribute a compiled fact. Phase 13 explainability uses recorded retrieval, temporal, graph, optimizer, and compiler evidence. Phase 14 inspection adds a structured, bounded read-only view over one such execution. Its optional comparison runs additional retrieval strategies explicitly. Neither mechanism sends a provider request. A logical request fingerprint is not a hash of provider wire bytes; downstream adapter transformations cannot be inferred from it. Unknown candidate or graph membership remains `NOT_AVAILABLE`.

Telemetry records model/provider identifiers, counts, timing, measurement source/tokenizer, status, and errors, not raw prompts/responses. A provider-reported input count and a local preflight/context count are different measurement bases and must not be added together. The dashboard groups by provider, model, and context measurement basis. See [RAG inspector](rag-inspector.md) and [explainability](explainability.md).

## Integrations and trust

MCP is optional STDIO with separately configured permissions. Local-file and JSON import connectors are explicit and credential-free; no connector is registered by default. FakeProvider is a simulation for tests/demo. Ollama and OpenAI-compatible local endpoints can be configured; external providers remain optional. The daemon is not a multi-user authorization boundary. Loopback binding, trusted local-user access, filesystem permissions, and disk encryption remain operational responsibilities.

No schema migration was added for Phases 14-18. The inspector, dashboards, benchmark, and demo use existing v7 data and temporary benchmark/demo databases.
