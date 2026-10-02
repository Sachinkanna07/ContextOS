# Phase 13 explainability audit

Audit performed against frozen `phase-12` HEAD `67a39c52f2f5b6e36dd78bb327c0b570a6ab6093`. The worktree was clean before this audit; no schema change appears necessary.

## Existing observable signals

- Phase 4 `HybridRetrievalEngine` has lexical and dense raw scores/ranks, reciprocal-rank fusion, bounded metadata adjustment, final score/rank, and stage counts. Its returned `strategy_results` contains only final top-k candidates, so the current trace cannot explain candidates rejected by temporal eligibility or final result limit.
- Phase 8 `GraphAugmentedRetrievalEngine` returns graph rank/score and bounded `GraphPath` records on selected results. Expansion already tracks seed nodes, traversed edge IDs, supporting memory IDs, and cycle-safe hop limits. Its public result loses some seed-to-memory path detail, and trace metadata contains internal graph IDs unsuitable as public output.
- Phase 7 memory records expose lifecycle status, temporal status/precision, observed/effective times, supersession links, slot identity, and provenance event ID. Temporal acceptance returns value-free transition decisions, but retrieval does not report why a result was ineligible.
- Phase 5 optimizer returns a per-candidate `CandidateDecision`: exact token cost, utility contributions, marginal utility, redundancy, selected state, exclusion code, and redundant-with ID. This signal is sufficient when captured from the same `SelectionResult`.
- Phase 6 compiler returns fact-level source memory IDs and provenance event IDs, input kind (including rescue), emitted/excluded facts and token costs, included memory IDs, token totals, and provenance coverage. It does not return a complete memory-to-fact transformation history for every deduplicated/merged/dropped candidate.
- Phase 9 model invocation receives a compiled context and persists aggregate telemetry. The terminal explanation should describe the context compiled for this execution; proving an actual external provider request requires a later model invocation tied to the same trace, which is not presently linked.
- Phase 10 MCP `explain` currently performs retrieval plus optimizer only and returns internal optimizer decisions with little temporal/compiler context. MCP policy checks `allow_read`; existing input/result limits are available.
- Phase 11 connector persistence keeps connector identity and source-item mappings. Memory `source_type` and `provenance_event_id` are safe summary fields; connector IDs and arbitrary source URIs/metadata require strict sanitization before exposure.
- Phase 12 `/compile` and `preview` perform retrieval, optimization, and compilation once, but discard intermediate retrieval and optimizer signals. CLI rendering already has `safe()` for untrusted strings.

## Gaps, identity, and proposed data flow

Signals are lost at API boundaries after retrieval, and compilation does not expose enough internal decision provenance to explain unobservable transformations. Introduce an ephemeral server-generated UUID trace ID at the explanation service entrypoint and pass it through one retrieval → optimization → compilation execution. Do not persist traces or add schema v8. Build the explanation response from those returned objects; do not replay stages to reconstruct reasons. Keep counts and arrays bounded, avoid raw query/content by default, and whitelist provenance fields.

The explainable candidate universe must be described honestly: candidates returned by retrieval and those explicitly rejected by its temporal/result-limit gates can be explained; the service cannot prove why corpus memories never returned by an index search were absent. A caller may request one memory ID; check that row directly and use the shared retrieval eligibility predicate for a provable temporal/lifecycle exclusion. Otherwise return `not_available`, without scanning the database.

## Privacy and phase boundary

Current memory content is private and source URI can contain usernames or local paths. Explanation defaults must contain only bounded IDs, enums, numeric signals, and sanitized provenance summaries. Include snippets/content only on explicit request. Do not forward graph database node/edge IDs as paths; return only verified labels, relation types, depth, and supporting memory IDs. No provider generation, LLM explanation, chain-of-thought, persistence, or Phase 14 inspector UI is in scope.

## Implementation acceptance constraints

Reuse actual retrieval, optimizer, and compiler objects. Add an API endpoint, Phase 12 `explain` command and `preview --explain`, and route MCP `contextos_explain_context` through the same service while preserving its existing top-level fields where practical. Add real SQLite pipeline tests, privacy/adversarial coverage, deterministic decision checks, and a synthetic 100/1000-memory benchmark. Document unsupported evidence explicitly.
