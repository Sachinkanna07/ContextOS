# ContextOS Changelog

All notable changes to ContextOS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-rc3] - 2026-10-03

- Stats and monitor omit absent optional model/provider filters and reject empty or whitespace-only filters. Dashboard API rejects blank filters too.
- Dashboard and model listing share concurrent, bounded inventory discovery (5 seconds per provider). Successful inventories cache for 10 seconds; failures/empty inventories refresh after 1 second. Concurrent refreshes coalesce per provider, provider replacement invalidates its cache, and failures do not suppress healthy providers.
- Ollama and OpenAI-compatible inventories use one request, bypass transient health-cache failures, and never fabricate default models for unavailable or empty inventories.
- Ollama 5xx diagnostics expose only recognized, bounded runtime facts and constrained exit codes; arbitrary error bodies, credentials, paths and terminal controls are excluded. Malformed chat/usage responses preserve typed exceptions; zero usage remains provider-reported and missing usage counts all dispatched message content on the appropriate token basis.
- Local Ollama 0.34.0 investigation reproduced a Qwen3.5 CUDA runner initialization failure (exit `0xc0000409`) with a minimal raw `/api/chat` request outside ContextOS. A subsequent ContextOS-shaped raw request succeeded but exhausted 256 output tokens on thinking with empty final content. Request semantics remain unchanged; no model-name workaround or automatic retry was added. See the RC3 release notes for validation scope.

## [1.0.0-rc2] - 2026-10-03

- Distribution renamed to `contextos-memory-runtime`; imports and commands remain `contextos`.
- Bounded startup serialization using cross-platform OS lifecycle locks (`msvcrt` on Windows, `flock` on POSIX) with configured timeouts (30s readiness, 45s lock).
- OS TCP listener ownership verification: verifies that the configured loopback listener belongs to the verified ContextOS daemon process tree, rejecting spoofed endpoint PID claims.
- Recoverable startup ownership: daemon publishes durable identity prior to binding, ensuring that readiness implies discoverable lifecycle registration even if the starter process exits prematurely.
- Compare-before-delete lifecycle cleanup preserving survivor state during concurrent start or failed attempts.

## [1.0.0-rc1] - 2026-10-02

### Added
- **Core Architecture & Persistence:** Local SQLite storage engine (Schema v7) with append-only event logging, memory lifecycle tracking, and deterministic local defaults. The runtime has Python package dependencies but needs no cloud account or model/tokenizer download with those defaults.
- **Privacy & Security Boundary:** `PatternSecretScanner` detects supported credential shapes across CLI, API, MCP, and connectors; policy rejects, quarantines, or redacts findings. Selected graph/provenance outputs suppress path-shaped labels and terminal sanitization removes control sequences; universal private-path redaction is not a scanner capability. Attack matrix verified against ANSI/OSC escape injection, zero-width chars, malformed JSONL, SQLite contention, and credential leakage.
- **Temporal Memory:** Slot-key state tracking supporting supersession, contradiction, and coexistence relations with optimized candidate peer lookup beyond 500 records.
- **Hybrid RAG & Retrieval Engine:** BM25 lexical indexing + dense vector search combined via Reciprocal Rank Fusion (RRF), with batched candidate hydration.
- **Token-aware Optimizer & Compiler:** Knapsack-based context selection under token budgets with fact/provenance evidence formatting and contextual framing.
- **Deterministic Memory Graph:** Graph projection engine supporting bounded path traversal, entity/relation links, dirty-state tracking, and opt-in graph-assisted retrieval.
- **Model Router & Telemetry:** Provider-agnostic router (`FakeProvider`, `Ollama`, `OpenAICompatible`) with per-invocation token telemetry and measurement basis provenance.
- **Model Context Protocol (MCP):** STDIO protocol implementation (`contextos-mcp`) with permission checks and bounded output schemas.
- **Connectors System:** Local file and JSON/JSONL connectors with incremental sync, content hashing, and skip invariants for unchanged files.
- **Terminal UX & Dashboard:** Interactive CLI (`contextos stats`, `monitor`, `desktop`), rich terminal formatting, and diagnostic health checks (`contextos doctor`).
- **Explainability & RAG Inspector:** Non-intrusive trace generation and context inspection CLI (`contextos inspect`) for single-pass and side-by-side retrieval mode comparison.
- **Canonical Benchmarking & Evaluation Suite:** Synthetic benchmarking framework (`contextos.benchmarks.final`) evaluating 100 and 1,000 memory corpora across Vector, Hybrid, Hybrid+Graph, and ContextOS strategies.
- **Packaging & Modular Distribution:** Core `contextos` package decoupled from heavyweight ML frameworks, making `sentence-transformers` an optional extra (`pip install "contextos-memory-runtime[embeddings]"`).

### Release blocker repairs
- Core defaults use hashed deterministic embeddings and word-token accounting labeled `APPROXIMATED`. Optional SentenceTransformer embeddings and exact tiktoken counting require explicit configuration.
- First-run demo needs no model/tokenizer assets or external network. Startup failure closes the initialized database before propagating the error.
- Wheel/sdist rules exclude local state and nested checkouts; `python tools/validate_release_artifacts.py` validates rebuilt archives.
- The canonical synthetic benchmark retains explicit `cl100k_base` counting; token reductions are not provider-billed savings.

### Known limitations
- Answer quality is **NOT VALIDATED**. Deterministic embeddings are an offline baseline, not a SentenceTransformer-quality claim.
- Graph retrieval stays opt-in because the current synthetic benchmark shows a ranking/latency tradeoff.
- Live providers have not been comprehensively validated.
- Latest independent cold CLI startup: approximately **2.27 seconds median** on Windows.
- Python 3.12 is declared but not validated; release validation uses Windows/Python 3.13.5.
