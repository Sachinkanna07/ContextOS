# ContextOS

**Local-first personal AI memory runtime.**

ContextOS runs locally as a background daemon and gives any connected LLM persistent, personalized memory — while minimizing the number of context tokens sent to that model.

## What This Is

- A **local memory runtime** — not a chatbot, not a RAG framework, not a cloud service.
- **Model-agnostic** — works with any LLM provider (OpenAI, Anthropic, Ollama, etc.).
- **Privacy-first** — all data stays local. Secrets are detected and blocked before storage.
- **Token-efficient** — retrieves and compiles only the most relevant context, measured and benchmarked.

## Status

**Phase 13 working tree** — Deterministic retrieval and compilation explanations (uncommitted).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Start the daemon
contextos start

# Ingest some context
contextos ingest "I prefer Python 3.12+ with strict type hints."
contextos ingest "I use pytest for testing and ruff for linting."

# Retrieve relevant memories
contextos retrieve "What are my coding preferences?"

# Compile optimized context for an LLM
contextos compile "Help me set up a new Python project" --show-context

# Explain retrieval and compilation decisions
contextos explain "What are my coding preferences?"

# Check system status
contextos status
contextos stats
```

## Terminal product

Start the daemon with `contextos start`, then use `contextos monitor` for a live
dashboard (`--model MODEL` filters one model, `--interval` accepts 0.5–60 seconds).
On Windows, `contextos desktop` opens the monitor in a separate terminal window.
The terminal uses the same loopback daemon and SQLite database.

```powershell
contextos health
contextos stats --model fake-default
contextos models list
contextos memories list
contextos memories search "coding preferences"
contextos memories show <memory-uuid>
"I prefer concise documentation." | contextos memories remember
contextos preview "Help with documentation" --budget 1000
contextos connectors list
contextos connectors status <connector-id>
contextos connectors sync <connector-id>
```

The dashboard and memory list show metadata without private memory text.
`memories show`, `preview --show-context`, and `--json` explicitly reveal content.
`memories remember` reads stdin or a hidden prompt so memory text stays out of
process arguments. Connector commands operate only on connectors registered in
the running daemon. No connector is registered by default.

Register local sources in `%LOCALAPPDATA%\contextos\config.toml` before starting
the daemon. The roots and import files must already exist; invalid configuration
stops startup. For example:

```toml
[connectors.local_files]
notes = ['C:\Users\me\Documents\notes']

[connectors.json_imports]
export = 'C:\Users\me\Documents\memory.jsonl'
```

Preflight and context counts use the target tokenizer or a labeled approximation.
Provider-reported counts are shown separately. Reduction is a comparison of
candidate and compiled context counted on the same model basis; it does not
claim better answer quality or fewer provider-billed tokens. Rows recorded
before schema v7 have unknown context token measurement provenance.

Run `python -m contextos.benchmarks.terminal` for local CLI, dashboard,
telemetry, search, and monitor polling latency measurements. These are runtime
measurements and contain no synthetic token-savings figures.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full engineering specification.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks
make check
```

## License

MIT
