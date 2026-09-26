# ContextOS

**Local-first personal AI memory runtime.**

ContextOS runs locally as a background daemon and gives any connected LLM persistent, personalized memory — while minimizing the number of context tokens sent to that model.

## What This Is

- A **local memory runtime** — not a chatbot, not a RAG framework, not a cloud service.
- **Model-agnostic** — works with any LLM provider (OpenAI, Anthropic, Ollama, etc.).
- **Privacy-first** — all data stays local. Secrets are detected and blocked before storage.
- **Token-efficient** — retrieves and compiles only the most relevant context, measured and benchmarked.

## Status

**Phase 1** — Core pipeline implementation.

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

# Check system status
contextos status
contextos stats
```

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
