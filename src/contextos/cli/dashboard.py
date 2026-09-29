"""Terminal dashboard rendering with no untrusted terminal control sequences."""

from __future__ import annotations

import re

from rich.console import Group
from rich.table import Table
from rich.text import Text


_escape = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\)|.)")


def safe(value: object, limit: int = 160, allow_newlines: bool = False) -> str:
    if value is None:
        return ""
    text = _escape.sub("", str(value))
    if allow_newlines:
        return "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t"))[:limit]
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def render_dashboard(data: dict, model: str | None = None) -> Group:
    counts = data["memories"]
    summary = data["summary"]
    models = summary.get("by_model", {})
    bases = data.get("context_measurement_bases", [])
    table = Table(title="ContextOS - Live activity", show_header=False)
    table.add_column("Metric", style="cyan", overflow="crop")
    table.add_column("Value", overflow="crop")
    table.add_row("Memories", f"{counts['active']} active | {counts['historical']} historical | {counts['expired']} expired")
    table.add_row("Connectors", ", ".join(f"{safe(c['id'])}: {safe(c['status'])}" for c in data["connectors"]) or "none registered")
    table.add_row("Models", ", ".join(
        f"{safe(m['model'], 32)} ({'unavailable' if not m['enabled'] else 'simulation' if m['simulated'] else 'local' if m['local'] else 'remote'})"
        for m in data.get("models", [])
    ) or "none discoverable")
    table.add_row("Successful invocations", str(summary["total_invocations"]))
    if (model or len(models) == 1) and len(bases) == 1 and bases[0]["source"] != "unknown":
        table.add_row("Context basis", f"{safe(bases[0]['source'])} | {safe(bases[0]['tokenizer'])}")
        table.add_row("Context avoided", f"{summary['total_tokens_avoided']:,} context tokens")
        table.add_row("Average reduction", f"{summary['average_reduction_ratio']:.1%} (arithmetic)")
        table.add_row("Weighted reduction", f"{summary['weighted_reduction_ratio']:.1%} (candidate-token weighted)")
    elif models:
        table.add_row("Reduction", "Select --model; a single known context-token basis is required")
    else:
        table.add_row("Reduction", "No measured invocations")
    activity = Table(title="Recent invocations - metadata only")
    for name in ("Time", "Model", "Status", "Preflight", "Provider input", "Candidate -> compiled", "Context source", "Avoided", "Graph", "Retrieve", "Compile"):
        activity.add_column(name, overflow="crop")
    for row in data["recent"]:
        activity.add_row(
            safe(row["timestamp"], 19), safe(row["model"], 32), safe(row["status"], 20),
            str(row["preflight_input_tokens"]),
            f"{row['provider_input_tokens']} ({safe(row['provider_measurement_source'], 20)})",
            f"{row['candidate_context_tokens']} -> {row['compiled_context_tokens']}",
            safe(row.get("context_measurement_source", "unknown"), 20),
            str(row["context_tokens_avoided"]), str(row["graph_expanded_count"]),
            f"{row['retrieval_ms']:.1f} ms", f"{row['compilation_ms']:.1f} ms",
        )
    if not data["recent"]:
        activity.add_row("No activity", "", "", "", "", "", "", "", "", "", "", "")
    return Group(table, activity, Text("Context and preflight counts use the recorded target tokenizer or approximation. Provider usage is separate; no cross-tokenizer totals are shown.", style="dim"))
