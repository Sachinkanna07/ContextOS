# Offline golden demo

Run `contextos demo --json` or `python -m contextos.demo` without starting a daemon, cloud account, API key, or local model server. It creates a temporary SQLite database, wires the normal deterministic local services, and removes the directory when finished. No real user data is read or modified.

The demo sends a preference and a later changed preference through ingestion and temporal acceptance, syncs a fake connector twice to show accepted then unchanged behavior, builds and traverses the graph, runs explainability and inspection without provider dispatch, invokes FakeProvider once, queries its telemetry, and confirms a secret-shaped input is rejected. The output reports measured IDs/counts/statuses and inspector token diff. FakeProvider response is simulated, not a production-provider proof. The synthetic demo is a walkthrough, not a benchmark or answer-quality assessment.
