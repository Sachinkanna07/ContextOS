# Phase 3 privacy data lifecycle

Phase 3 is a local, deterministic privacy boundary. Pattern detection reduces
risk but cannot guarantee detection of every secret.

1. **Raw input in memory:** The caller and `IngestRequest` hold raw input only
   for the duration of the request. The privacy gate scans it before any event,
   candidate, memory, index, or log write. Request content, source URI, and tags
   are excluded from normal model representations, and API validation responses
   remove attacker-controlled `input` and `ctx` fields.
2. **Raw input after scanning:** No component intentionally retains it. A
   process-local keyed fingerprint may be retained for short-lived correlation.
3. **Sanitized text:** Allow-mode text or redacted text is the only input passed
   to extraction and the only content eligible for raw-event persistence.
4. **Secret findings:** Persisted findings contain category, severity, offsets,
   detector name, confidence, a category-only placeholder, and a one-way
   keyed fingerprint. The key is not persisted, and findings never contain the
   matched value.
5. **Provenance:** Content, source type, source URI, and tags are scanned.
   Provenance stores sanitized content, safe source identifiers,
   source trust classification, and the value-free privacy assessment.
6. **Candidate memories:** Candidates are scanned again across content, evidence,
   source fields, tags, and nested metadata. Returned candidates contain only
   sanitized fields and their privacy assessment.
7. **Rejected candidates:** Rejected candidate text is not persisted. The result
   may contain a value-free blocked assessment for diagnostics.

`skip_secret_scan` is retained only for API compatibility and cannot bypass the
persistence gate. Strict mode rejects before content persistence. Redact mode
continues with sanitized text. Warn mode quarantines sanitized input and returns
no candidates.

External pages, imported documents, tool results, and model output are always
treated as data. Source trust is descriptive metadata. Instruction-like content
from untrusted sources is blocked from becoming a memory candidate; it cannot
change privacy policy or application control flow.
