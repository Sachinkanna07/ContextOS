# ContextOS explainability

Phase 13 exposes an ephemeral, deterministic account of one retrieval, optimizer, and compiler execution. The final context is what ContextOS prepared for a downstream model. Standalone explanations and previews mark provider dispatch as `NOT_ATTEMPTED`; actual provider dispatch evidence is captured only when a real model invocation executes through `ModelService.ask(..., explain=True)`. The explanation trace has a server-generated UUID; timings and trace IDs can vary while ranking and decision fields remain deterministic for unchanged inputs and configuration.

## What the trace can establish

### 1. Retrieval & Channels
For memories returned by the retrieval stage, the trace reports lexical BM25 rank and raw score, dense rank and raw score, fused score, reciprocal rank contributions, metadata adjustments, rank, and source strategies. BM25 and dense scores use different scales; compare ranks/fused scores rather than interpreting one as a calibrated percentage.

Candidate absence is categorized truthfully without guessing:
- `SELECTED`: Memory was retrieved, selected by the optimizer, and included in the compiled context.
- `COMPILER_EXCLUDED`: Memory was selected by the optimizer but the compiler's persisted-in-memory `included_memory_ids` evidence does not include it.
- `RETRIEVED_BUT_OPTIMIZER_EXCLUDED`: Memory was retrieved, but optimizer excluded it due to token budget or redundancy constraints.
- `RESULT_LIMIT`: Memory was present in the fused/reranked candidate set before the final top-K truncation cut it.
- `CHANNEL_NOT_RETRIEVED`: Memory is indexed and eligible, but was not surfaced by bounded candidate channels for this query.
- `INDEX_NOT_PRESENT`: Memory is directly proven absent from both lexical BM25 and dense vector indices, and graph retrieval was not active for this explanation. Graph-channel membership is not inferred from those two stores.
- `TEMPORALLY_INELIGIBLE`: Stored memory state or temporal scope policy excluded the candidate.
- `NOT_AVAILABLE`: Evidence is insufficient to establish why the memory was absent.

Channel candidate IDs and pre-limit IDs are retained with explicit truncation flags. If a requested memory is outside a truncated snapshot, the explanation returns `NOT_AVAILABLE`; it does not guess between channel absence and result-limit exclusion.

### 2. Graph Evidence
Graph traces capture structured path nodes and edges collected during graph traversal without second-pass querying:
- Traversal bounds: strictly bounded to `max_hops <= 3` (up to 3 edges and up to 4 nodes per path), `max_nodes <= 100`, and `max_edges <= 250`.
- `GraphPathNode`: `node_id`, `node_type`, safe sanitized label, and `project_scope` (if stored). Missing labels are reported as `null` or excluded; labels are never invented.
- `GraphPathEdge`: `edge_type`, `confidence`, `supporting_memory_ids`, and `project_scope`.
- `GraphCandidateEvidence`: `seed_memory_id`, `candidate_memory_id`, `hop_count`, `graph_score`, `path_nodes`, `path_edges`, and `scope_match` evidence.

### 3. Temporal Evidence
The structured `TemporalEvidenceResolver` reports provable persisted state and relations:
- Lifecycle status (`active`, `superseded`, `expired`, `deleted`, `purged`).
- Stored relations (`CORRECTS`, `COEXISTS_WITH`, `CONTRADICTS`, `SUPERSEDES`).
- Related endpoint state (`present`, `deleted`, or `missing`) resolved for either direction of a persisted relation.
- Replacement memory ID (`replacement_memory_id`) when superseded.
- Observed and effective timestamps.
- Acceptance rationale is strictly `null` (no fictional historical reasoning or motives are inferred).

### 4. Compiler transformations
Compiler actions are recorded from emitted and excluded compiler facts: `RAW_INCLUDED`, `DEDUPLICATED`, `MERGED`, `COMPRESSED`, `RESCUED_FROM_OVERSIZED_MEMORY`, and `DROPPED`. Exact raw inclusion, merge source count, rescue input kind, and exclusion reason are evidenced. When output differs from the original memory but the compiler does not record whether it was clause extraction or compression, the trace reports `UNKNOWN` rather than claiming `COMPRESSED`. The trace does not invent a transformation for a memory with no compiler fact; optimizer selection and compiler inclusion are reported separately.

### 5. Provider Dispatch Proof & Receipt
For standalone explanations (`contextos explain`, MCP `contextos_explain_context`, `POST /api/v1/explain`), the dispatch state is strictly:
- `ProviderDispatchState.NOT_ATTEMPTED`: "prepared by ContextOS; no provider dispatch attempted"

For real model invocations (`ModelService.ask`), evidence tracks:
- `state`:
  - `REQUEST_CONSTRUCTED`: Context and ModelRequest constructed and fingerprinted.
  - `DISPATCH_ATTEMPTED`: Provider invocation `generate()` was attempted at the adapter boundary (does not claim verified wire delivery).
  - `RESPONSE_RECEIVED`: Downstream provider returned a successful response.
  - `DISPATCH_FAILED`: Provider invocation raised an error or timed out. Evidence is attached to the raised Python exception and returned in the structured `/ask` API error response when available. SQLite telemetry records only generic `status="error"`, not the receipt.
- `compiled_context_sha256`: SHA-256 fingerprint of the exact compiled context text.
- `logical_request_sha256`: SHA-256 fingerprint of the actual `ModelRequest` fields observed immediately before provider invocation, represented as system prompt, compiled context, and user prompt. Provider adapters subsequently transform this into provider-specific payloads; this is not a wire-payload hash.
- `context_match`: Boolean proof confirming the actual downstream `ModelRequest` object carried the exact compiled ContextOS context string before hashing.
- `preflight_input_tokens`: Preflight token count calculated before dispatch.
- `provider_input_tokens`: Token count returned directly by the downstream provider (kept separate from ContextOS context tokens).

## Privacy

Responses omit raw prompt text, private credentials, source URIs, connector metadata, and filesystem paths by default. Provenance includes only whitelisted source categories and event UUIDs; unknown source values are replaced with `unknown`. Terminal output is sanitized against ANSI CSI, OSC titles, OSC hyperlinks, carriage returns, backspaces, nulls, and bidi overrides. Traces are ephemeral (never persisted), and no schema v8 migration is required.

## Use

```powershell
contextos explain "What are my Python preferences?" --budget 1000 --mode hybrid --limit 25
contextos explain "What are my Python preferences?" --json
contextos explain "What are my Python preferences?" --show-content
contextos explain "What are my Python preferences?" --memory-id 12345678-1234-4234-8234-123456789012
contextos explain "What did I previously prefer?" --temporal-scope historical
contextos preview "What are my Python preferences?" --explain
```

The API accepts `POST /api/v1/explain` with `query`, `mode`, `budget`, `limit`, `graph`, `temporal_scope`, `include_content`, and optional `target_memory_id`. Query length is capped at 10,000 characters, budget at 8,000 tokens, and candidates at 100.

