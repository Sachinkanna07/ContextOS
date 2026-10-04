# ContextOS

ContextOS is a local-first, model-independent AI memory runtime. It is not a chatbot. It accepts activity through a privacy boundary, extracts candidate facts, resolves temporal state, persists memories in SQLite, and supplies bounded context to a chosen model. A local daemon exposes a CLI and API; optional STDIO MCP and explicitly configured connectors use the same memory pipeline.

## Project status

**Core product status: feature-complete for the current scope.**

The current public prerelease is **[v1.0.0-rc4](https://github.com/Sachinkanna07/ContextOS/releases/tag/v1.0.0-rc4)**. It remains marked as a prerelease while the project gets real-world soak time; that does **not** mean the core ContextOS architecture is unfinished. Package-index publication and artifact-identity verification are treated as release-distribution gates and are reported separately from source validation.

The completed core includes persistent local-first memory, privacy filtering before persistence, temporal lifecycle handling, hybrid retrieval and reranking, token-aware context selection, the Context Compiler, optional graph-assisted retrieval, provider routing, Ollama/local-model integration, native OpenAI, Anthropic, and Gemini text adapters, configurable OpenAI-compatible endpoints, MCP/connectors, telemetry, explainability, diagnostics, CLI/API surfaces, and packaged releases.

RC4 source validation completed with **813 tests passed, 0 failed, 0 skipped**, clean isolated package checks, real Ollama/Qwen memory-aware inference, telemetry/dashboard checks, and a final review with **0 Critical, 0 High, and 0 Medium** findings. Ollama is **LIVE VALIDATED**. OpenAI, Anthropic, and Gemini are **IMPLEMENTED + CONTRACT TESTED**; **LIVE VALIDATION NOT PERFORMED** for those cloud providers.

### Scope boundary

ContextOS is a **memory and context runtime**, not a model provider and not a chatbot. Native adapters for **OpenAI/GPT, Anthropic/Claude, and Google Gemini are implemented and contract-tested in RC4**. Ollama is live-validated; live validation of those cloud adapters has not been performed. Remote providers remain opt-in and require explicit remote-provider consent.

Broader live-provider coverage and additional provider adapters should be treated as post-1.0 expansion, not as evidence that the current ContextOS core is incomplete.

## 30-second overview

- **What it does:** stores and retrieves useful long-term context for LLM applications.
- **Core ideas:** hybrid retrieval, temporal memory, token-aware context selection, privacy boundaries, and explainability.
- **Runs locally:** SQLite-backed, CLI/API driven, with optional MCP and connector support.
- **Why it matters:** reduces reliance on raw conversation history and makes context selection more explicit and inspectable.
- **Current status:** core product feature-complete for the current scope; `v1.0.0-rc4` is the current public prerelease. Answer-quality gains are not claimed as validated.

## Why it exists

Long conversation histories are expensive and can carry stale or contradictory facts. ContextOS combines lexical and dense retrieval, temporal eligibility, a token-aware selector, and a context compiler. It records what it selected and what it could prove, without claiming that fewer context tokens automatically improve answers or provider billing.

## Quick start

ContextOS keeps one memory store across supported local and cloud providers. Native text adapters cover Ollama, OpenAI, Anthropic, and Gemini; named OpenAI-compatible endpoints are configurable. Remote providers are disabled by default. **A remote request transmits its compiled ContextOS memory context to the selected provider.** See [provider setup and privacy](docs/providers.md).

Python 3.12+ is declared; this release has been verified on Windows with Python 3.13. Python 3.12 is not yet validated. Core defaults and the offline demo require no commercial API key, model download, or tokenizer cache.

### Installation

Core lightweight installation (uses deterministic/local capabilities):
```powershell
pip install contextos-memory-runtime
```

Core defaults use hashed deterministic embeddings and word-token accounting labeled `APPROXIMATED`. These embeddings are an offline baseline; SentenceTransformer retrieval quality is not implied. FakeProvider is a local simulation.

Run `python tools/validate_release_artifacts.py` after building release archives to check package contents.
Canonical RC4 artifacts are **236,598 bytes** for the wheel and **336,852 bytes** for the sdist. SHA256: wheel `1d31c30c6cbaa713b37cfc33a695e43462140e05ef05430b1574446bcb5c1e26`; sdist `1f7143f0bb879071eb3ee30d5cedf4c31c75462f8c3ab323bb7192d80b00a3af`. These sizes exclude installed dependencies. Run `python tools/validate_release_artifacts.py` after a rebuild for exact validation.

With optional local sentence-transformers embeddings extra:
```powershell
pip install "contextos-memory-runtime[embeddings]"
```

Installing the extra does not change defaults. Explicitly set `[embedding]` with `model = "all-MiniLM-L6-v2"` in your ContextOS `config.toml` to select it; first use may download that model unless provisioned locally. Exact context counting is also opt-in: set `[token_counter]` with `encoding = "cl100k_base"` or `"o200k_base"`, and provision its tokenizer cache before offline use. Default `encoding = "deterministic"` needs no tokenizer assets.

For development:
```powershell
pip install -e ".[dev]"
```

### Usage

```powershell
contextos demo
contextos start
contextos doctor
contextos stats
"I prefer concise technical explanations." | contextos memories remember
contextos inspect "What explanation style do I prefer?"
contextos ask "What explanation style do I prefer?" --provider ollama --model qwen2.5-coder:7b
contextos models providers
contextos stop
```

The daemon listens on loopback by default. Review your local configuration before changing its host, connector roots, MCP permissions, or provider settings. Prefer `memories remember` for private text: a command-line argument may be visible in process lists and shell history.

## Architecture and memory lifecycle

Input flows through privacy scanning, extraction, temporal acceptance, and SQLite persistence. Retrieval reads the memory store through ephemeral BM25 and dense indexes. The graph is a separate deterministic projection. The optimizer selects within a token budget; the compiler turns selected memories into context with fact/provenance evidence. The router sends the prepared request to a configured local or optional remote provider. Graph-assisted retrieval remains opt-in because its ranking performance is mixed in the local benchmark.

See [architecture](docs/architecture.md), [explainability](docs/explainability.md), and [RAG inspector](docs/rag-inspector.md) for exact boundaries. Explainability and inspection describe ContextOS preparation, not an unseen provider wire payload. Provider dispatch is `NOT_ATTEMPTED` during inspection.

## Privacy and integrations

Secret scanning does not universally redact private filesystem paths. Selected graph/provenance output surfaces suppress path-shaped labels; terminal sanitization removes control sequences. These are separate output protections.

Secrets are scanned before accepted memory is stored; the legacy `--skip-secret-scan` request field does not disable that boundary. Metadata dashboards omit memory bodies, prompts, source paths, and credentials by default. Explicit content-view commands can reveal private text on your terminal; use them deliberately. Local storage is not a substitute for OS disk encryption or trusted local-user access.

Credential-free local file and JSON/JSONL connectors must be configured explicitly; none are registered by default. See [connectors](docs/connectors.md). MCP is disabled by default, uses STDIO, and separates read, write, and telemetry permissions; see [MCP](docs/mcp.md). Ollama and compatible local endpoints are first-class options; external providers are optional. FakeProvider supports deterministic tests/demo and is identified as simulated.

## Terminal product

| Command | Purpose |
| --- | --- |
| `contextos status`, `health`, `doctor` | Daemon state and non-destructive diagnostics |
| `contextos stats [--model ID] [--provider ID] [--compare] [--today/--week]` | Measured activity by provider/model and token basis |
| `contextos monitor [--model ID] [--provider ID]` | Poll the same bounded local dashboard |
| `contextos inspect "query" [--mode hybrid] [--graph] [--memory UUID] [--compare] [--json]` | Retrieval-to-compiler decision evidence |
| `contextos explain "query"`, `preview "query"` | Explain or preview context without provider dispatch |
| `contextos graph stats/search/show` | Projection statistics and bounded graph paths |
| `contextos memories current/conflicts/history` | Temporal metadata; content is opt-in |
| `contextos connectors list/status/sync` | Registered connector state and explicit sync |
| `contextos models list`, `contextos telemetry` | Provider inventory and measured invocation records |
| `contextos ask "question" --provider ollama --model MODEL` | Ask a model with compiled memory context; `--show-context` opts into printing private context |
| `contextos models providers` | Safe provider configuration, credential-presence, and discovery status |
| `contextos benchmark [--extended]`, `contextos demo` | Isolated synthetic evaluation and offline walkthrough |

The dashboard groups provider and model together and keeps context-token counts separate from provider-reported usage. A reduction bar appears only when a single known context tokenizer basis can be compared. Older telemetry with unknown provenance is not silently combined. `--today` is the current UTC day; `--week` is a rolling seven-day window. Session-wide history is not yet persisted as a distinct aggregate.
Connector status includes currently tracked source-item count; the existing schema does not retain last-sync accepted/unchanged/failed totals for historical display.

## Evidence and limits

The [benchmark](docs/benchmarking.md) compares full history, vector, hybrid, hybrid+graph, and compiled ContextOS context on ten fixed questions at 100 and 1,000 synthetic memories. It reports relevance ranking and context tokens, not answer quality. The [security](docs/security.md), [performance](docs/performance.md), [demo](docs/demo.md), and [release checklist](docs/release-checklist.md) documents separate measured, simulated, and unverified claims. A 5,000-memory run is opt-in. No cloud deployment or commercial-provider proof is implied by the local test suite.

Answer quality is **NOT VALIDATED**. Live providers have not been comprehensively validated. Graph retrieval stays opt-in because the current synthetic benchmark shows a recall/latency tradeoff. Latest independent cold CLI startup was approximately **2.27 seconds median** on Windows; this is a local measurement, not a guarantee.

## Development

```powershell
python -m pytest tests -q -ra
python -m compileall -q src tests
python -m contextos.benchmarks.final
python -m pip check
git diff --check
```

`make test`, `make lint`, `make typecheck`, and `make check` are available when `make` is installed. The package is currently version 1.0.0-rc4. License: MIT.

Background `contextos start` uses the configured readiness timeout (30 seconds by default in RC4) before reporting startup failure. Existing-daemon readiness requires the health PID to match the verified recorded PID at the configured host and port. Startup and stop share a Windows/POSIX lifecycle lock with a 45-second default acquisition timeout; concurrent starters wait, then report the verified daemon as already running. The persistent `contextos.lock` file is reusable and OS lock ownership is released on process exit. PID publication is atomic, and failed startup cleans up only its own process tree and matching PID state. `contextos doctor` can be run immediately after a successful start.
