"""ContextOS CLI — main application and top-level commands.

The CLI is a thin client. All business logic runs in the daemon.
Commands communicate with the daemon via HTTP (localhost).
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Optional

import httpx
import typer
from rich.console import Console

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

app.add_typer(memory_app, name="memory")
app.add_typer(config_app, name="config")

console = Console()
error_console = Console(stderr=True)


def _base_url() -> str:
    """Get the daemon base URL."""
    from contextos.config.settings import load_settings
    settings = load_settings()
    return f"http://{settings.daemon.host}:{settings.daemon.port}"


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    """Make an API request to the daemon."""
    url = f"{_base_url()}/api/v1{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(method, url, **kwargs)
            if response.status_code >= 400:
                try:
                    error = response.json()
                    error_console.print(f"[red]Error:[/red] {error.get('error', response.text)}")
                except Exception:
                    error_console.print(f"[red]Error:[/red] {response.text}")
                raise typer.Exit(1)
            return response
    except httpx.ConnectError:
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
    from contextos.daemon.manager import start_daemon, is_running

    settings = load_settings()
    running, pid = is_running(settings)

    if running:
        console.print(f"[yellow]ContextOS daemon is already running (PID: {pid})[/yellow]")
        raise typer.Exit(0)

    if foreground:
        console.print("[bold]Starting ContextOS daemon (foreground)...[/bold]")
        start_daemon(settings, foreground=True)
    else:
        console.print("[bold]Starting ContextOS daemon...[/bold]")
        start_daemon(settings, foreground=False)
        console.print(f"[green]✓[/green] Daemon started on {settings.daemon.host}:{settings.daemon.port}")


@app.command()
def stop() -> None:
    """Stop the ContextOS daemon."""
    from contextos.config.settings import load_settings
    from contextos.daemon.manager import stop_daemon

    try:
        stop_daemon(load_settings())
        console.print("[green]✓[/green] Daemon stopped")
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
        from contextos.core.models import SystemStatus
        from contextos.cli.formatters import format_status
        format_status(SystemStatus(**data))


@app.command()
def stats(
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Show token statistics."""
    resp = _api("GET", "/stats")
    data = resp.json()

    if json_output:
        console.print_json(json.dumps(data))
    else:
        from contextos.core.models import TokenStats
        from contextos.cli.formatters import format_stats
        format_stats(TokenStats(**data))


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
    source_uri: Optional[str] = typer.Option(None, "--file", help="Source file path"),
    memory_type: Optional[str] = typer.Option(None, "--type", "-t", help="Memory type hint"),
    skip_scan: bool = typer.Option(False, "--skip-secret-scan", help="Skip secret detection"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
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
        from contextos.core.models import IngestResult
        from contextos.cli.formatters import format_ingest_result
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
        from contextos.core.models import RetrievalResult
        from contextos.cli.formatters import format_retrieval_result
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
        from contextos.core.models import CompiledContext
        from contextos.cli.formatters import format_compiled_context
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
    status_filter: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    type_filter: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
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
        from contextos.core.models import Memory
        from contextos.cli.formatters import format_memory_list
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
        from contextos.core.models import RetrievalResult
        from contextos.cli.formatters import format_retrieval_result
        format_retrieval_result(RetrievalResult(**data))


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
        from contextos.core.models import Memory
        from contextos.cli.formatters import format_memory
        format_memory(Memory(**data), detailed=True)


@memory_app.command("delete")
def memory_delete(
    memory_id: str = typer.Argument(..., help="Memory ID"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Soft-delete a memory."""
    if not force:
        confirm = typer.confirm(f"Delete memory {memory_id[:8]}…?")
        if not confirm:
            raise typer.Abort()

    _api("DELETE", f"/memories/{memory_id}")
    console.print(f"[green]✓[/green] Memory {memory_id[:8]}… deleted")


@memory_app.command("purge")
def memory_purge(
    memory_id: str = typer.Argument(..., help="Memory ID"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Hard-delete a memory (irreversible)."""
    if not force:
        confirm = typer.confirm(
            f"[red]PERMANENTLY[/red] purge memory {memory_id[:8]}…? This cannot be undone.",
            default=False,
        )
        if not confirm:
            raise typer.Abort()

    _api("DELETE", f"/memories/{memory_id}/purge")
    console.print(f"[green]✓[/green] Memory {memory_id[:8]}… purged")


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
