"""Rich output formatters for the ContextOS CLI.

Centralizes all terminal formatting so CLI commands stay clean.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contextos.core.enums import MemoryStatus, PrivacyLevel
from contextos.core.models import (
    CompiledContext,
    IngestResult,
    Memory,
    RetrievalResult,
    ScoredMemory,
    SystemStatus,
    TokenStats,
)

console = Console()
error_console = Console(stderr=True)


# --- Status Colors ---

STATUS_COLORS: dict[MemoryStatus, str] = {
    MemoryStatus.ACTIVE: "green",
    MemoryStatus.CANDIDATE: "yellow",
    MemoryStatus.SUPERSEDED: "dim",
    MemoryStatus.CONTRADICTED: "red",
    MemoryStatus.EXPIRED: "dim yellow",
    MemoryStatus.HISTORICAL: "dim",
    MemoryStatus.DELETED: "dim red",
    MemoryStatus.PURGED: "dim red",
    MemoryStatus.MERGED: "dim cyan",
}

PRIVACY_COLORS: dict[PrivacyLevel, str] = {
    PrivacyLevel.PUBLIC: "green",
    PrivacyLevel.PERSONAL: "blue",
    PrivacyLevel.SENSITIVE: "yellow",
    PrivacyLevel.RESTRICTED: "red",
}


def format_status(status: SystemStatus) -> None:
    """Print system status."""
    table = Table(title="ContextOS Status", show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    running_text = Text("● Running", style="bold green") if status.daemon_running else Text("● Stopped", style="bold red")
    table.add_row("Status", running_text)
    table.add_row("PID", str(status.pid or "-"))
    table.add_row("Uptime", _format_duration(status.uptime_seconds))
    table.add_row("", "")
    table.add_row("Memories", f"{status.active_memories} active / {status.total_memories} total")
    table.add_row("Events", str(status.total_events))
    table.add_row("", "")
    table.add_row("Embedding Model", status.embedding_model or "-")
    table.add_row("Vector Index", f"{status.vector_index_size} vectors")
    table.add_row("BM25 Index", f"{status.bm25_index_size} documents")
    table.add_row("Database Size", _format_bytes(status.database_size_bytes))
    table.add_row("Data Directory", status.data_directory)

    console.print(table)


def format_stats(stats: TokenStats) -> None:
    """Print token statistics."""
    table = Table(title="Token Statistics", show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Tokens Stored", f"{stats.total_tokens_stored:,}")
    table.add_row("Tokens Per Memory", f"{stats.tokens_per_memory:.1f}")
    table.add_row("", "")
    table.add_row("Total Compilations", f"{stats.total_compilations:,}")
    table.add_row("Total Tokens Compiled", f"{stats.total_tokens_compiled:,}")
    table.add_row("Total Tokens Saved", f"{stats.total_tokens_saved:,}")
    table.add_row("Avg Compression Ratio", f"{stats.average_compression_ratio:.1%}")

    console.print(table)


def format_memory(memory: Memory, detailed: bool = False) -> None:
    """Print a single memory."""
    status_color = STATUS_COLORS.get(memory.status, "white")

    header = f"[bold]{memory.type.value.upper()}[/bold] │ "
    header += f"[{status_color}]{memory.status.value}[/{status_color}]"
    header += f" │ confidence: {memory.confidence:.0%}"
    header += f" │ importance: {memory.importance:.0%}"

    panel = Panel(
        memory.content,
        title=header,
        subtitle=f"ID: {str(memory.id)[:8]}… │ {memory.token_count} tokens │ {memory.created_at.strftime('%Y-%m-%d %H:%M')}",
        border_style=status_color,
        padding=(0, 1),
    )
    console.print(panel)

    if detailed:
        detail_table = Table(show_header=False, box=None, padding=(0, 2))
        detail_table.add_column("Key", style="dim")
        detail_table.add_column("Value")

        detail_table.add_row("Full ID", str(memory.id))
        detail_table.add_row("Source", f"{memory.source_type}" + (f" ({memory.source_uri})" if memory.source_uri else ""))
        detail_table.add_row("Privacy", Text(memory.privacy_level.value, style=PRIVACY_COLORS.get(memory.privacy_level, "white")))
        detail_table.add_row("Tags", ", ".join(memory.tags) if memory.tags else "-")
        detail_table.add_row("Access Count", str(memory.access_count))
        detail_table.add_row("Last Accessed", memory.last_accessed_at.strftime('%Y-%m-%d %H:%M') if memory.last_accessed_at else "-")
        detail_table.add_row("Version", str(memory.version))
        if memory.expires_at:
            detail_table.add_row("Expires", memory.expires_at.strftime('%Y-%m-%d %H:%M'))
        if memory.superseded_by:
            detail_table.add_row("Superseded By", str(memory.superseded_by)[:8] + "…")
        if memory.supersedes:
            detail_table.add_row("Supersedes", str(memory.supersedes)[:8] + "…")

        console.print(detail_table)


def format_memory_list(memories: list[Memory]) -> None:
    """Print a list of memories as a table."""
    if not memories:
        console.print("[dim]No memories found.[/dim]")
        return

    table = Table(title=f"{len(memories)} Memories")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Content", max_width=60)
    table.add_column("Type", width=12)
    table.add_column("Status", width=12)
    table.add_column("Conf.", width=6, justify="right")
    table.add_column("Tokens", width=7, justify="right")
    table.add_column("Created", width=12)

    for mem in memories:
        status_color = STATUS_COLORS.get(mem.status, "white")
        content_preview = mem.content[:57] + "…" if len(mem.content) > 57 else mem.content

        table.add_row(
            str(mem.id)[:8] + "…",
            content_preview,
            mem.type.value,
            Text(mem.status.value, style=status_color),
            f"{mem.confidence:.0%}",
            str(mem.token_count),
            mem.created_at.strftime("%Y-%m-%d"),
        )

    console.print(table)


def format_ingest_result(result: IngestResult) -> None:
    """Print ingestion result."""
    if result.memories_created:
        console.print(f"[green]✓[/green] Created {len(result.memories_created)} memor{'y' if len(result.memories_created) == 1 else 'ies'}")
        for mid in result.memories_created:
            console.print(f"  [dim]{str(mid)[:8]}…[/dim]")

    if result.memories_merged:
        console.print(f"[cyan]↗[/cyan] Merged into {len(result.memories_merged)} existing memor{'y' if len(result.memories_merged) == 1 else 'ies'}")

    if result.secrets_detected:
        if result.secrets_redacted:
            console.print("[yellow]⚠ Secrets detected and redacted[/yellow]")
        else:
            console.print("[yellow]⚠ Secrets detected[/yellow]")

    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")


def format_retrieval_result(result: RetrievalResult, show_trace: bool = False) -> None:
    """Print retrieval results."""
    if not result.memories:
        console.print("[dim]No memories found.[/dim]")
        return

    console.print(f"\n[bold]Retrieved {len(result.memories)} memories[/bold] for: [italic]{result.query}[/italic]\n")

    for i, sm in enumerate(result.memories, 1):
        score_text = f"score: {sm.final_score:.4f}"
        if sm.vector_score is not None:
            score_text += f" (vec: {sm.vector_score:.3f}"
        if sm.bm25_score is not None:
            score_text += f", bm25: {sm.bm25_score:.3f}"
        if sm.vector_score is not None or sm.bm25_score is not None:
            score_text += ")"

        status_color = STATUS_COLORS.get(sm.memory.status, "white")
        console.print(
            f"  [{status_color}]{i}.[/{status_color}] {sm.memory.content}"
        )
        console.print(f"     [dim]{score_text} │ {sm.memory.type.value} │ {sm.memory.token_count} tokens[/dim]")

    if show_trace:
        console.print("\n[bold]Pipeline Trace[/bold]")
        trace_table = Table(box=None, padding=(0, 1))
        trace_table.add_column("Stage", style="cyan")
        trace_table.add_column("In", justify="right")
        trace_table.add_column("Out", justify="right")
        trace_table.add_column("Latency", justify="right")

        for stage in result.trace.stages:
            trace_table.add_row(
                stage.stage_name,
                str(stage.input_count),
                str(stage.output_count),
                f"{stage.latency_ms:.1f}ms",
            )

        trace_table.add_row(
            "[bold]Total[/bold]", "", str(result.trace.total_results),
            f"[bold]{result.trace.total_latency_ms:.1f}ms[/bold]",
        )
        console.print(trace_table)


def format_compiled_context(compiled: CompiledContext, show_context: bool = False) -> None:
    """Print compilation result."""
    console.print(f"\n[bold]Context Compilation[/bold] for: [italic]{compiled.query}[/italic]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Budget", f"{compiled.budget:,} tokens")
    table.add_row("Compiled", f"{compiled.total_tokens:,} tokens")
    table.add_row("Compression", f"{compiled.compression_ratio:.1%}")
    table.add_row("Memories Included", f"{compiled.memories_included} / {compiled.memories_considered}")

    console.print(table)

    if show_context:
        console.print("\n[bold]Compiled Context:[/bold]")
        console.print(Panel(compiled.context_text, border_style="green"))


def format_doctor_results(results: dict) -> None:
    """Print doctor diagnostic results."""
    console.print(f"\n[bold]ContextOS Doctor[/bold] (v{results.get('version', '?')})\n")

    for check_name, check_result in results.get("checks", {}).items():
        ok = check_result.get("ok", False)
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {icon} {check_name}")

        for key, value in check_result.items():
            if key == "ok":
                continue
            if key == "warnings" and value:
                for w in value:
                    console.print(f"      [yellow]⚠ {w}[/yellow]")
            elif key != "warnings":
                console.print(f"      [dim]{key}: {value}[/dim]")

    overall = results.get("overall", False)
    console.print()
    if overall:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print("[bold red]Some checks failed.[/bold red]")


# --- Utilities ---


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _format_bytes(bytes_val: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024  # type: ignore
    return f"{bytes_val:.1f} TB"
