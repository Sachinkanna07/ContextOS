"""ContextOS CLI — main application and top-level commands.

The CLI is a thin client. All business logic runs in the daemon.
Commands communicate with the daemon via HTTP (localhost).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from contextos import __version__

app = typer.Typer(
    name="contextos",
    help="ContextOS — Local-first personal AI memory runtime",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# Sub-command groups
memory_app = typer.Typer(help="Manage memories", no_args_is_help=True)
config_app = typer.Typer(help="Manage configuration", no_args_is_help=True)
memories_app = typer.Typer(help="Manage memories", no_args_is_help=True)
connectors_app = typer.Typer(help="Inspect and sync registered connectors", no_args_is_help=True)
models_app = typer.Typer(help="Inspect registered models", no_args_is_help=True)
graph_app = typer.Typer(help="Inspect the bounded memory graph", no_args_is_help=True)

app.add_typer(memory_app, name="memory")
app.add_typer(config_app, name="config")
app.add_typer(memories_app, name="memories")
app.add_typer(connectors_app, name="connectors")
app.add_typer(models_app, name="models")
app.add_typer(graph_app, name="graph")

console = Console()
error_console = Console(stderr=True)


def _base_url() -> str:
    """Get the daemon base URL."""
    from contextos.config.settings import load_settings
    settings = load_settings()
    host = "127.0.0.1" if settings.daemon.host == "localhost" else settings.daemon.host
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}:{settings.daemon.port}"


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    """Make an API request to the daemon."""
    url = f"{_base_url()}/api/v1{path}"
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.request(method, url, **kwargs)
            if response.status_code >= 400:
                try:
                    from contextos.cli.dashboard import safe
                    error = response.json()
                    error_msg = safe(error.get("error", error.get("detail", "Request failed")), limit=500)
                    error_console.print("Error: ", Text(error_msg))
                except Exception:
                    error_console.print("Error: Request failed")
                raise typer.Exit(1)
            return response
    except (httpx.ConnectError, httpx.TimeoutException):
        error_console.print("[red]Error:[/red] Cannot connect to ContextOS daemon.")
        error_console.print("Start it with: [bold]contextos start[/bold]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Top-level Commands
# ---------------------------------------------------------------------------


@app.command()
def start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
) -> None:
    """Start the ContextOS daemon."""
    from contextos.config.settings import load_settings
    from contextos.core.exceptions import DaemonAlreadyRunningError, DaemonLockTimeoutError
    from contextos.daemon.manager import start_daemon

    settings = load_settings()
    if foreground:
        import os
        from contextos.config.settings import Settings
        child_settings = os.environ.get("CONTEXTOS_DAEMON_SETTINGS")
        if child_settings:
            settings = Settings.model_validate_json(child_settings)
        console.print("[bold]Starting ContextOS daemon (foreground)...[/bold]")
        start_daemon(settings, foreground=True)
        return

    console.print("[bold]Starting ContextOS daemon...[/bold]")
    try:
        start_daemon(settings, foreground=False)
    except DaemonAlreadyRunningError as exc:
        console.print(f"[yellow]ContextOS daemon is already running (PID: {exc.pid})[/yellow]")
        raise typer.Exit(0)
    except DaemonLockTimeoutError as exc:
        error_console.print(f"Error: {exc}")
        raise typer.Exit(1) from exc
    except (RuntimeError, OSError) as exc:
        error_console.print(f"Error: {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green][OK][/green] Daemon started on {settings.daemon.host}:{settings.daemon.port}")


@app.command()
def stop() -> None:
    """Stop the ContextOS daemon."""
    from contextos.config.settings import load_settings
    from contextos.daemon.manager import stop_daemon

    try:
        stop_daemon(load_settings())
        console.print("[green][OK][/green] Daemon stopped")
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show system status."""
    resp = _api("GET", "/status")
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_status
        from contextos.core.models import SystemStatus
        format_status(SystemStatus(**data))


@app.command()
def stats(
    model: str | None = typer.Option(None, "--model", help="Filter by model ID"),
    provider: str | None = typer.Option(None, "--provider", help="Filter by provider ID"),
    compare: bool = typer.Option(False, "--compare", help="Show provider/model breakdown"),
    today: bool = typer.Option(False, "--today", help="Use the current UTC day"),
    week: bool = typer.Option(False, "--week", help="Use the last seven days"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show real activity and model-specific token statistics."""
    if today and week:
        raise typer.BadParameter("Choose --today or --week")
    resp = _api("GET", "/dashboard", params={
        "model": model, "provider": provider,
        "period": "today" if today else "week" if week else "all",
    })
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.dashboard import render_dashboard
        console.print(render_dashboard(data, model, compare=compare))


@app.command()
def monitor(
    model: str | None = typer.Option(None, "--model", help="Filter by model ID"),
    provider: str | None = typer.Option(None, "--provider", help="Filter by provider ID"),
    interval: float = typer.Option(2.0, "--interval", min=0.5, max=60.0),
    samples: int | None = typer.Option(None, "--samples", min=1, max=1000),
) -> None:
    """Watch bounded local activity until Ctrl-C or the requested sample count."""
    from contextos.cli.dashboard import render_dashboard
    count = 0
    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            while samples is None or count < samples:
                data = _api("GET", "/dashboard", params={"model": model, "provider": provider}).json()
                live.update(render_dashboard(data, model), refresh=True)
                count += 1
                if samples is None or count < samples:
                    time.sleep(interval)
    except KeyboardInterrupt:
        return


@app.command()
def desktop() -> None:
    """Open the local monitor in its own Windows terminal window."""
    if sys.platform != "win32":
        monitor()
        return
    import subprocess
    subprocess.Popen(
        [sys.executable, "-m", "contextos", "monitor"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        close_fds=True,
    )


@app.command()
def health(json_output: bool = typer.Option(False, "--json")) -> None:
    """Check daemon and database/index health."""
    result = _api("POST", "/doctor").json()
    if json_output:
        console.print_json(json.dumps(result))
    else:
        from contextos.cli.formatters import format_doctor_results
        format_doctor_results(result)
    if not result.get("overall"):
        raise typer.Exit(1)


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Run diagnostic checks."""
    resp = _api("POST", "/doctor")
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_doctor_results
        format_doctor_results(data)


@app.command()
def ingest(
    content: str = typer.Argument(..., help="Content to ingest"),
    source: str = typer.Option("cli_input", "--source", "-s", help="Source type"),
    source_uri: str | None = typer.Option(None, "--file", help="Source file path"),
    memory_type: str | None = typer.Option(None, "--type", "-t", help="Memory type hint"),
    skip_scan: bool = typer.Option(False, "--skip-secret-scan", help="Deprecated compatibility flag; privacy scanning is always enforced"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
) -> None:
    """Ingest content into ContextOS."""
    # Handle file input
    if source_uri:
        from pathlib import Path
        file_path = Path(source_uri)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            source = "file"
        else:
            error_console.print(f"[red]Error:[/red] File not found: {source_uri}")
            raise typer.Exit(1)

    payload = {
        "content": content,
        "source_type": source,
        "source_uri": source_uri,
        "skip_secret_scan": skip_scan,
    }
    if memory_type:
        payload["memory_type"] = memory_type
    if tags:
        payload["tags"] = [t.strip() for t in tags.split(",")]

    resp = _api("POST", "/ingest", json=payload)
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_ingest_result
        from contextos.core.models import IngestResult
        format_ingest_result(IngestResult(**data))


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query to retrieve memories for"),
    trace: bool = typer.Option(False, "--trace", help="Show pipeline trace"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Max results per strategy"),
) -> None:
    """Retrieve relevant memories."""
    payload = {
        "query": query,
        "config": {"vector_top_k": top_k, "bm25_top_k": top_k},
    }

    resp = _api("POST", "/retrieve", json=payload)
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_retrieval_result
        from contextos.core.models import RetrievalResult
        format_retrieval_result(RetrievalResult(**data), show_trace=trace)


@app.command()
def compile(
    query: str = typer.Argument(..., help="Query/task to compile context for"),
    show_context: bool = typer.Option(False, "--show-context", help="Show compiled context"),
    budget: int = typer.Option(4000, "--budget", "-b", help="Token budget"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    format: str = typer.Option("text", "--format", help="Output format (text/json)"),
) -> None:
    """Compile optimized context for an LLM."""
    payload = {
        "query": query,
        "config": {"budget": budget, "format": format},
    }

    resp = _api("POST", "/compile", json=payload)
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_compiled_context
        from contextos.core.models import CompiledContext
        format_compiled_context(CompiledContext(**data), show_context=show_context)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"ContextOS v{__version__}")


# ---------------------------------------------------------------------------
# Memory Sub-commands
# ---------------------------------------------------------------------------


@memory_app.command("list")
def memory_list(
    status_filter: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    type_filter: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """List memories."""
    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter
    if type_filter:
        params["type"] = type_filter

    resp = _api("GET", "/memories", params=params)
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_memory_list
        from contextos.core.models import Memory
        memories = [Memory(**m) for m in data]
        format_memory_list(memories)


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Search memories."""
    payload = {"query": query, "config": {"vector_top_k": limit, "bm25_top_k": limit}}
    resp = _api("POST", "/retrieve", json=payload)
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_retrieval_result
        from contextos.core.models import RetrievalResult
        format_retrieval_result(RetrievalResult(**data))


@memories_app.command("list")
def memories_list(
    status_filter: str | None = typer.Option(None, "--status"),
    limit: int = typer.Option(25, "--limit", min=1, max=100),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List bounded memory metadata; content requires --json or show."""
    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter
    rows = _api("GET", "/memories", params=params).json()
    if json_output:
        console.print_json(json.dumps(rows))
    else:
        from rich.table import Table

        from contextos.cli.dashboard import safe
        table = Table(title="Memories - metadata only")
        for column in ("ID", "Type", "Status", "Privacy", "Tokens"):
            table.add_column(column)
        for row in rows:
            table.add_row(safe(row["id"], 36), safe(row["type"]), safe(row["status"]),
                          safe(row["privacy_level"]), str(row["token_count"]))
        console.print(table)


@memories_app.command("search")
def memories_search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", min=1, max=50),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Search memories; show IDs and scores by default."""
    result = _api("POST", "/retrieve", json={"query": query,
                 "config": {"vector_top_k": limit, "bm25_top_k": limit}}).json()
    if json_output:
        console.print_json(json.dumps(result))
    else:
        from rich.table import Table

        from contextos.cli.dashboard import safe
        table = Table(title="Memory matches - metadata only")
        for column in ("ID", "Type", "Score"):
            table.add_column(column)
        for item in result["memories"][:limit]:
            table.add_row(safe(item["memory"]["id"], 36), safe(item["memory"]["type"]),
                          f"{item['final_score']:.3f}")
        console.print(table)


@memories_app.command("show")
def memories_show(memory_id: str = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:
    """Show one memory, including its private content."""
    from uuid import UUID
    try:
        memory_id = str(UUID(memory_id))
    except ValueError:
        raise typer.BadParameter("Expected a memory UUID") from None
    row = _api("GET", f"/memories/{memory_id}").json()
    if json_output:
        console.print_json(json.dumps(row))
    else:
        from contextos.cli.dashboard import safe
        console.print(Text(safe(row["content"], 10_000, allow_newlines=True)))
        console.print(f"ID: {memory_id} | {safe(row['status'])} | {safe(row['privacy_level'])}")


@memories_app.command("remember")
def memories_remember() -> None:
    """Remember text from stdin or a hidden prompt; never place it in process arguments."""
    import getpass
    content = sys.stdin.read(10_001) if not sys.stdin.isatty() else getpass.getpass("Memory: ")
    if not content.strip() or len(content) > 10_000:
        raise typer.BadParameter("Memory must contain 1-10,000 characters")
    result = _api("POST", "/remember", json={"text": content}).json()
    console.print(f"Accepted {result['count']} memories")


@connectors_app.command("list")
def connectors_list() -> None:
    """List registered connectors and their persisted state."""
    rows = _api("GET", "/connectors").json()
    from rich.table import Table

    from contextos.cli.dashboard import safe
    table = Table(title="Connectors")
    for column in ("ID", "Status", "Enabled", "Error code"):
        table.add_column(column)
    for row in rows:
        table.add_row(safe(row["id"]), safe(row["status"]), str(row["enabled"]), safe(row["error_code"] or ""))
    if not rows:
        table.add_row("No connectors registered", "", "", "")
    console.print(table)


@connectors_app.command("status")
def connectors_status(connector_id: str = typer.Argument(...)) -> None:
    """Show one connector state."""
    from contextos.cli.dashboard import safe
    rows = _api("GET", "/connectors").json()
    row = next((item for item in rows if item["id"] == connector_id), None)
    if row is None:
        error_console.print("Connector is not registered")
        raise typer.Exit(1)
    for key in ("id", "status", "enabled", "error_code", "last_success_at"):
        console.print(f"{key}: {safe(row.get(key))}")


@connectors_app.command("sync")
def connectors_sync(connector_id: str = typer.Argument(...)) -> None:
    """Run a registered connector through its existing sync pipeline."""
    from contextos.cli.dashboard import safe
    result = _api("POST", f"/connectors/{safe(connector_id, 100)}/sync").json()
    console.print(f"{safe(result['status'])}: {result['accepted']} accepted, {result['failed']} failed")
    if result["status"] not in ("success", "disabled"):
        raise typer.Exit(1)


@models_app.command("list")
def models_list(json_output: bool = typer.Option(False, "--json")) -> None:
    """List currently discoverable models."""
    rows = _api("GET", "/models").json()
    if json_output:
        console.print_json(json.dumps(rows))
    else:
        from rich.table import Table

        from contextos.cli.dashboard import safe
        table = Table(title="Discoverable models")
        for column in ("Provider", "Model", "Local"):
            table.add_column(column)
        for row in rows:
            table.add_row(safe(row.get("provider_id")), safe(row.get("model_id")), str(row.get("local")))
        if not rows:
            table.add_row("No models available", "", "")
        console.print(table)


@app.command()
def preview(query: str = typer.Argument(...), budget: int = typer.Option(4000, "--budget", min=1, max=32000),
            show_context: bool = typer.Option(False, "--show-context"),
            explain: bool = typer.Option(False, "--explain")) -> None:
    """Preview selected memories and provenance without invoking a model."""
    from contextos.cli.dashboard import safe
    if explain:
        result = _api("POST", "/explain", json={"query": query, "budget": min(budget, 8000), "include_content": show_context}).json()
        _print_explanation(result)
        return
    result = _api("POST", "/compile", json={"query": query, "config": {"budget": budget}}).json()
    console.print(f"Compiled {result['total_tokens']} / {result['budget']} tokens")
    console.print(f"Selected {result['memories_included']} of {result['memories_considered']} memories")
    for fact in result["facts"][:50]:
        ids = ", ".join(safe(item, 36) for item in fact["source_memory_ids"])
        console.print(Text(f"{safe(fact['fact_id'])}: {ids} | {safe(fact['input_kind'])}"))
    if result["excluded_facts"]:
        console.print("Exclusions:")
        for fact in result["excluded_facts"][:50]:
            console.print(Text(f"{safe(fact['fact_id'])}: {safe(fact['reason'])}"))
    if show_context:
        console.print(Text(safe(result["context_text"], 50_000, allow_newlines=True)))


def _print_explanation(result: dict) -> None:
    from contextos.services.explainability import safe_text
    console.print(f"QUERY TRACE {safe_text(result.get('trace_id'), 36)}")
    for row in result.get("candidates", [])[:100]:
        retrieval = row.get("retrieval", {})
        temporal = row.get("temporal", {})
        optimizer = row.get("optimizer", {})
        console.print(f"[{row.get('rank', '?')}] Memory {safe_text(row.get('memory_id'), 36)}")
        console.print(f"    retrieved: {safe_text(retrieval.get('origin'))} rank {row.get('rank')}")
        console.print(f"    temporal: {safe_text(temporal.get('status'))}; eligible={temporal.get('eligible')}")
        graph_paths = row.get("graph", [])
        graph_labels = []
        for gp in graph_paths:
            for pn in gp.get("path_nodes", []):
                if pn.get("label"):
                    graph_labels.append(pn["label"])
        label_str = f"; entities={','.join(graph_labels[:3])}" if graph_labels else ""
        console.print(f"    graph: {len(graph_paths)} path(s); tokens={row.get('token_cost')}{label_str}")
        console.print(f"    optimizer: {'selected' if row.get('selected') else 'excluded'}; reason={safe_text(optimizer.get('reason'))}")
        if row.get("content") is not None:
            console.print(Text(safe_text(row["content"], 1000)))
    if result.get("content") is not None:
        console.print("Compiled context:")
        console.print(Text(safe_text(result["content"], 20_000)))
    if result.get("requested_memory") is not None:
        console.print("Requested memory: " + json.dumps(result["requested_memory"], ensure_ascii=True))
    dispatch = result.get("provider_dispatch", {})
    state = dispatch.get("state", "NOT_ATTEMPTED")
    console.print(f"Provider dispatch: {safe_text(state, 32)}")
    console.print("Final context: prepared by ContextOS")
    console.print("Final context stats: " + json.dumps(result.get("final_context", {}), ensure_ascii=True))


@app.command()
def explain(query: str = typer.Argument(...), budget: int = typer.Option(1000, "--budget", min=1, max=8000),
            mode: str = typer.Option("hybrid", "--mode"), graph: bool = typer.Option(True, "--graph/--no-graph"),
            limit: int = typer.Option(25, "--limit", min=1, max=100),
            memory_id: str | None = typer.Option(None, "--memory-id", help="Explain one requested memory when it was observed"),
            temporal_scope: str = typer.Option("current", "--temporal-scope"),
            json_output: bool = typer.Option(False, "--json"),
            show_content: bool = typer.Option(False, "--show-content")) -> None:
    """Explain retrieval, selection, and compilation using recorded pipeline signals."""
    result = _api("POST", "/explain", json={"query": query, "budget": budget, "mode": mode,
                                             "graph": graph, "limit": limit,
                                             "target_memory_id": memory_id,
                                             "temporal_scope": temporal_scope,
                                             "include_content": show_content}).json()
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=True))
    else:
        _print_explanation(result)


@app.command()
def inspect(
    query: str = typer.Argument(...),
    mode: str = typer.Option("hybrid", "--mode"),
    graph: bool = typer.Option(False, "--graph/--no-graph"),
    budget: int = typer.Option(1000, "--budget", min=1, max=8000),
    limit: int = typer.Option(25, "--limit", min=1, max=100),
    memory: str | None = typer.Option(None, "--memory"),
    target_model: str | None = typer.Option(None, "--target-model"),
    compare: bool = typer.Option(False, "--compare"),
    show_content: bool = typer.Option(False, "--show-content"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect one bounded context preparation, with optional retrieval comparison."""
    result = _api("POST", "/inspect", json={
        "query": query, "mode": mode, "graph": graph, "budget": budget,
        "limit": limit, "target_memory_id": memory, "target_model": target_model,
        "compare": compare, "include_content": show_content,
    }).json()
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=True))
        return
    from rich.table import Table

    from contextos.services.explainability import safe_text
    console.print(Text("ContextOS RAG inspection  " + safe_text(result["inspection_id"], 36)))
    stages = Table(title="Pipeline")
    for name in ("Stage", "Input", "Output", "Removed", "Latency ms"):
        stages.add_column(name)
    for stage in result["stages"][:20]:
        stages.add_row(
            safe_text(stage["name"], 64), str(stage["input_count"]),
            str(stage["output_count"]), str(stage["removed_count"]),
            f"{stage['latency_ms']:.2f}",
        )
    console.print(stages)
    diff = result["context_diff"]
    context = Table(title="Context token diff")
    context.add_column("Basis")
    context.add_column("Tokens")
    for label, key in (("Retrieved", "candidate_tokens"), ("Optimized", "optimized_tokens"),
                       ("Compiled", "compiled_tokens"), ("Removed", "tokens_removed")):
        context.add_row(label, str(diff[key]))
    console.print(context)
    console.print(Text(
        f"Reduction {diff['reduction_ratio']:.1%} [{safe_text(diff['token_measurement_source'])}; "
        f"{safe_text(diff['tokenizer'], 80)}] | facts {diff['facts_emitted']} emitted, "
        f"{diff['facts_excluded']} excluded | provider {safe_text(result['provider_dispatch']['state'])}"
    ))
    candidates = Table(title="Candidates (bounded)")
    for name in ("Rank", "Memory", "Origin", "BM25", "Dense", "Graph", "Temporal", "Selected", "Compiler"):
        candidates.add_column(name, overflow="crop")
    for row in result["candidates"][:limit]:
        retrieval = row["retrieval"]
        candidates.add_row(
            str(row["rank"]), safe_text(row["memory_id"], 36),
            safe_text(retrieval["origin"], 20), str(retrieval["lexical_rank"] or "-"),
            str(retrieval["dense_rank"] or "-"), str(retrieval["graph_rank"] or "-"),
            safe_text(row["temporal"]["status"], 20), str(row["selected"]),
            safe_text(",".join(row["compiler_transformations"]) or "not available", 50),
        )
    console.print(candidates)
    if result.get("requested_memory"):
        requested = result["requested_memory"]
        console.print(Text(
            f"Memory {safe_text(requested['memory_id'], 36)}: "
            f"{safe_text(requested['reason_code'], 50)} ({safe_text(requested['reason'], 80)})"
        ))
    for row in result["candidates"][:limit]:
        for path in row["graph"][:5]:
            nodes = [safe_text(node.get("label") or node["node_type"], 120) for node in path["path_nodes"]]
            edges = [safe_text(edge["edge_type"], 40) for edge in path["path_edges"]]
            route = "".join(f" --{edge}--> {nodes[index + 1]}" for index, edge in enumerate(edges)
                            if index + 1 < len(nodes))
            if nodes:
                console.print(Text(f"Graph {safe_text(row['memory_id'], 36)}: {nodes[0]}{route}"))
    if result.get("comparison"):
        table = Table(title="Retrieval comparison (no ground truth)")
        for name in ("Mode", "Candidates", "Overlap", "Latency ms"):
            table.add_column(name)
        for run in result["comparison"]["runs"]:
            table.add_row(safe_text(run["mode"]), str(run["candidate_count"]),
                          str(run["overlap_with_inspection"]), f"{run['latency_ms']:.2f}")
        console.print(table)
    if result.get("content") is not None:
        console.print(Text(safe_text(result["content"], 20_000)))


@app.command()
def benchmark(
    extended: bool = typer.Option(False, "--extended"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run the isolated local synthetic comparison benchmark."""
    import asyncio

    from contextos.benchmarks.final import measure
    result = asyncio.run(measure(extended=extended))
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=True))
        return
    from rich.table import Table
    table = Table(title=result["label"])
    for column in ("Memories", "Strategy", "Recall@5", "MRR", "NDCG@10", "Reduction"):
        table.add_column(column)
    for size, corpus in result["corpora"].items():
        for name, row in corpus["strategies"].items():
            table.add_row(size, name, f"{row['retrieval']['recall@5']:.3f}",
                          f"{row['retrieval']['mrr']:.3f}",
                          f"{row['retrieval']['ndcg@10']:.3f}",
                          f"{row['weighted_token_reduction']:.1%}" if row["weighted_token_reduction"] is not None else "UNKNOWN")
    console.print(table)
    console.print("[BENCHMARKED] Synthetic queries only; answer quality NOT AVAILABLE")


@app.command()
def demo(json_output: bool = typer.Option(False, "--json")) -> None:
    """Run a disposable offline walkthrough with deterministic services."""
    import asyncio

    from contextos.demo import run_demo
    result = asyncio.run(run_demo())
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=True))
        return
    from rich.table import Table
    table = Table(title=result["label"])
    table.add_column("Stage")
    table.add_column("Observed result")
    table.add_row("Temporal", f"{result['temporal']['previous_status']} -> {result['temporal']['current_status']}")
    table.add_row("Connector", f"{result['connector_first']['accepted']} accepted; {result['connector_second']['unchanged']} unchanged on repeat")
    table.add_row("Graph", f"{result['graph']['nodes']} nodes, {result['graph']['edges']} edges, {result['graph']['expansion_candidates']} expansion candidates")
    table.add_row("Inspector", f"{result['inspection']['candidates']} candidates, {result['inspection']['context_diff']['facts_emitted']} facts")
    table.add_row("Model", f"{result['model']['provider']}/{result['model']['model']}: {result['model']['dispatch_state']}")
    table.add_row("Privacy", "secret rejected" if result["privacy_secret_rejected"] else "secret was not rejected")
    console.print(table)
    console.print("Synthetic data was stored in a temporary directory and removed after the run.")


@graph_app.command("stats")
def graph_stats_cli(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show persisted graph projection counts and freshness."""
    data = _api("GET", "/graph/stats").json()
    if json_output:
        console.print_json(json.dumps(data, ensure_ascii=True))
        return
    from rich.table import Table
    table = Table(title="Memory graph projection")
    table.add_column("Metric")
    table.add_column("Measured value")
    for label, key in (("Nodes", "nodes"), ("Edges", "edges"), ("Supports", "supports")):
        table.add_row(label, str(data[key]))
    table.add_row("Projection", "dirty" if data["dirty"] else "clean")
    table.add_row("Average total degree", str(data["average_total_degree"]) if data["average_total_degree"] is not None else "UNKNOWN")
    console.print(table)


def _print_graph_paths(data: dict[str, Any]) -> None:
    from contextos.services.explainability import safe_text
    console.print(Text(f"Graph candidates: {data['candidate_count']}"))
    for candidate in data["candidates"][:10]:
        console.print(Text(f"Memory {safe_text(candidate['memory_id'], 36)} score {candidate['graph_score']:.4f}"))
        for path in candidate["paths"][:5]:
            nodes = [safe_text(node.get("label") or node["node_type"], 120) for node in path["path_nodes"]]
            edges = [safe_text(edge["edge_type"], 40) for edge in path["path_edges"]]
            route = "".join(f" --{edge}--> {nodes[index+1]}" for index, edge in enumerate(edges)
                            if index + 1 < len(nodes))
            if nodes:
                console.print(Text(nodes[0] + route))


@graph_app.command("search")
def graph_search_cli(entity: str = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:
    """Traverse from an entity using the existing bounded graph engine."""
    data = _api("GET", "/graph/search", params={"entity": entity}).json()
    if json_output:
        console.print_json(json.dumps(data, ensure_ascii=True))
    else:
        _print_graph_paths(data)


@graph_app.command("show")
def graph_show_cli(memory_id: str = typer.Argument(...), json_output: bool = typer.Option(False, "--json")) -> None:
    """Traverse graph paths from one memory ID."""
    from uuid import UUID
    try:
        safe_id = str(UUID(memory_id))
    except ValueError:
        raise typer.BadParameter("memory_id must be a UUID") from None
    data = _api("GET", f"/graph/show/{safe_id}").json()
    if json_output:
        console.print_json(json.dumps(data, ensure_ascii=True))
    else:
        _print_graph_paths(data)


def _print_temporal_rows(data: dict[str, Any]) -> None:
    from rich.table import Table

    from contextos.services.explainability import safe_text
    rows = data.get("history", data.get("memories", []))
    table = Table(title="Temporal memory metadata")
    for name in ("Memory", "Lifecycle", "Temporal", "Observed", "Effective", "Replaced by"):
        table.add_column(name, overflow="crop")
    content_lines = []
    for row in rows[:50]:
        table.add_row(
            safe_text(row["memory_id"], 36), safe_text(row["lifecycle"], 20),
            safe_text(row["temporal_status"], 20), safe_text(row["observed_at"], 32),
            safe_text(row["effective_at"], 32), safe_text(row["superseded_by"], 36),
        )
        if row.get("content") is not None:
            content_lines.append((safe_text(row["memory_id"], 36), safe_text(row["content"], 1000)))
    console.print(table)
    for memory_id, content in content_lines:
        console.print(Text(f"{memory_id}: {content}"))
    if data.get("relations"):
        for relation in data["relations"][:50]:
            console.print(Text(
                f"{safe_text(relation['relation_type'], 32)}: "
                f"{safe_text(relation['related_memory_id'], 36)} "
                f"({safe_text(relation['related_memory_state'], 20)})"
            ))


@memories_app.command("current")
def memories_current(limit: int = typer.Option(25, "--limit", min=1, max=50),
                     json_output: bool = typer.Option(False, "--json")) -> None:
    """List current memory metadata without private content."""
    data = _api("GET", "/temporal/current", params={"limit": limit}).json()
    console.print_json(json.dumps(data, ensure_ascii=True)) if json_output else _print_temporal_rows(data)


@memories_app.command("conflicts")
def memories_conflicts(limit: int = typer.Option(25, "--limit", min=1, max=50),
                       json_output: bool = typer.Option(False, "--json")) -> None:
    """List contradicted memory metadata without private content."""
    data = _api("GET", "/temporal/conflicts", params={"limit": limit}).json()
    console.print_json(json.dumps(data, ensure_ascii=True)) if json_output else _print_temporal_rows(data)


@memories_app.command("history")
def memories_history(memory_id: str = typer.Argument(...),
                     show_content: bool = typer.Option(False, "--show-content"),
                     json_output: bool = typer.Option(False, "--json")) -> None:
    """Inspect a bounded slot timeline and persisted relation evidence."""
    from uuid import UUID
    try:
        safe_id = str(UUID(memory_id))
    except ValueError:
        raise typer.BadParameter("memory_id must be a UUID") from None
    data = _api("GET", f"/temporal/history/{safe_id}",
                params={"include_content": show_content}).json()
    console.print_json(json.dumps(data, ensure_ascii=True)) if json_output else _print_temporal_rows(data)


@memory_app.command("inspect")
def memory_inspect(
    memory_id: str = typer.Argument(..., help="Memory ID"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Inspect a memory's full details."""
    resp = _api("GET", f"/memories/{memory_id}")
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.cli.formatters import format_memory
        from contextos.core.models import Memory
        format_memory(Memory(**data), detailed=True)


@memory_app.command("delete")
def memory_delete(
    memory_id: str = typer.Argument(..., help="Memory ID"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Soft-delete a memory."""
    if not force:
        confirm = typer.confirm(f"Delete memory {memory_id[:8]}...?")
        if not confirm:
            raise typer.Abort()

    _api("DELETE", f"/memories/{memory_id}")
    console.print(f"[green][OK][/green] Memory {memory_id[:8]}... deleted")


@memory_app.command("purge")
def memory_purge(
    memory_id: str = typer.Argument(..., help="Memory ID"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Hard-delete a memory (irreversible)."""
    if not force:
        confirm = typer.confirm(
            f"[red]PERMANENTLY[/red] purge memory {memory_id[:8]}...? This cannot be undone.",
            default=False,
        )
        if not confirm:
            raise typer.Abort()

    _api("DELETE", f"/memories/{memory_id}/purge")
    console.print(f"[green][OK][/green] Memory {memory_id[:8]}... purged")


# ---------------------------------------------------------------------------
# Config Sub-commands
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    from contextos.config.settings import load_settings
    settings = load_settings()
    console.print_json(settings.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
