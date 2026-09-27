# ContextOS MCP

ContextOS offers a local-first [Model Context Protocol](https://modelcontextprotocol.io/)
adapter over STDIO. It is disabled until explicitly enabled and has no network
listener. Run it with `contextos-mcp`; STDOUT is reserved solely for MCP JSON-RPC
traffic and diagnostics use stderr.

```toml
[mcp]
enabled = true
transport = "stdio"
allow_read = true
allow_write = false
allow_telemetry = true
```

## Tools and permissions

Read permission enables `contextos_search_memory`, `contextos_compile_context`,
`contextos_current_state`, `contextos_memory_history`, `contextos_graph_neighbors`,
and `contextos_explain_context`. Telemetry permission enables
`contextos_telemetry_summary`. Write permission enables only
`contextos_remember`. There is no MCP delete, purge, raw database, filesystem,
configuration, provider, model-routing, or credential tool.

`remember` treats MCP input as untrusted. It runs the normal privacy gate,
candidate extraction, classification, and temporal acceptance pipeline before
durable persistence. It never calls a repository create operation on supplied
text. The MCP tools do not automatically invoke a language-model provider.

Each accepted candidate uses the existing Phase 7 temporal transaction. A
multi-candidate request therefore has explicit per-candidate transactional
semantics: if a later candidate fails, the response is `PARTIAL_WRITE` and
identifies the already accepted memory IDs. It is not represented as success.

## Limits and privacy

Inputs are bounded (10,000 characters by default); search, history, graph hops
(maximum three), graph nodes/edges, traces, and compilation budgets are also
bounded. Invalid values fail with a stable error code. Results expose evidence
metadata and provenance IDs, but never source URIs, raw scanner findings,
credentials, database paths, stack traces, or environment values.
The write interface intentionally does not accept arbitrary source metadata;
the `metadata` argument is rejected at the MCP boundary.

MCP request telemetry is intentionally process-local and bounded. It records
only request ID, optional UUID session ID, tool name, timestamp, latency,
success/error code, and aggregate counts. It excludes arguments, queries,
remember text, context, raw responses, and secrets. A persistent telemetry table
would require a schema migration without offering useful local-STDIO retention,
so Phase 10 does not add one.
