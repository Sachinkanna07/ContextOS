# ContextOS connectors

Phase 11 supports credential-free local activity sources: explicitly configured
text/Markdown files, JSON/JSONL imports, and a deterministic fake connector for
tests. A connector returns bounded `ConnectorItem` values; it cannot write a
memory row. The manager sends every item through privacy, extraction, temporal
acceptance, persistence, and normal graph/index invalidation.

Source identity is `(connector_id, external_id)` plus revision/content hash.
Unchanged items are skipped before ingestion. Cursor and source identity state
are persisted without raw source content. Failed items stop cursor advancement;
the next sync retries from the prior safe cursor.

Local files are restricted to configured resolved roots, an extension allowlist
(`.txt`, `.md`, `.json`, `.jsonl`), deterministic ordering, UTF-8 decoding, and
a file-size limit. Symlink targets outside an allowed root are excluded.

Source deletion records the source item as deleted. The default policy keeps
derived memories: deleting a source is not a user request to purge memory.
Disabling a connector stops sync and retains state and memories. Future policy
may explicitly expire source-bound memory; automatic purge is never performed.

Connector input, metadata, Markdown, HTML-like text, and URLs are untrusted
data. Privacy scanner findings, raw content, credentials, and secret values are
not stored in cursor state or connector telemetry.
