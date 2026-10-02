"""Terminal dashboard rendering with no untrusted terminal control sequences."""

from __future__ import annotations

import re
from typing import Any

from rich.console import Group, RenderableType
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


def _model_kind(model: dict[str, Any]) -> str:
    if not model["enabled"]:
        return "unavailable"
    if model["simulated"]:
        return "simulation"
    return "local" if model["local"] else "remote"


def render_dashboard(
    data: dict[str, Any], model: str | None = None, compare: bool = False
) -> Group:
    counts = data["memories"]
    summary = data["summary"]
    models = summary.get("by_model", {})
    bases = data.get("context_measurement_bases", [])
    table = Table(title="ContextOS - Live activity", show_header=False)
    table.add_column("Metric", style="cyan", overflow="crop")
    table.add_column("Value", overflow="crop")
    table.add_row(
        "Memories",
        f"{counts['active']} active | {counts['historical']} historical | "
        f"{counts['expired']} expired",
    )
    temporal = data.get("temporal")
    if temporal:
        table.add_row(
            "Temporal",
            f"{temporal['superseded']} superseded | {temporal['contradicted']} contradicted",
        )
    table.add_row(
        "Connectors",
        ", ".join(
            f"{safe(c['id'])}: {safe(c['status'])} ({c.get('tracked_items', 0)} tracked)"
            for c in data["connectors"]
        ) or "none registered",
    )
    mcp = data.get("mcp")
    if mcp:
        table.add_row(
            "MCP config", "disabled" if not mcp["enabled"] else
            f"configured {safe(mcp['transport'])} | read={mcp['read']} | write={mcp['write']}"
        )
    table.add_row(
        "Models",
        ", ".join(
            f"{safe(item['model'], 32)} ({_model_kind(item)})"
            for item in data.get("models", [])
        ) or "none discoverable",
    )
    table.add_row("Successful invocations", str(summary["total_invocations"]))
    error_count = sum(row.get("errors", 0) for row in data.get("provider_models", []))
    table.add_row("Recorded errors", str(error_count))
    if (model or len(models) == 1) and len(bases) == 1 and bases[0]["source"] != "unknown":
        table.add_row(
            "Context basis",
            f"{safe(bases[0]['source'])} | {safe(bases[0]['tokenizer'])}",
        )
        table.add_row("Context avoided", f"{summary['total_tokens_avoided']:,} context tokens")
        table.add_row("Average reduction", f"{summary['average_reduction_ratio']:.1%} (arithmetic)")
        table.add_row(
            "Weighted reduction",
            f"{summary['weighted_reduction_ratio']:.1%} (candidate-token weighted)",
        )
    elif models:
        table.add_row("Reduction", "Select --model; a single known context-token basis is required")
    else:
        table.add_row("Reduction", "No measured invocations")
    activity = Table(title="Recent invocations - metadata only")
    for name in (
        "Time", "Model", "Status", "Preflight", "Provider input", "Candidate -> compiled",
        "Lex/Dense/Selected", "Context source", "Avoided", "Graph", "Retrieve", "Compile",
        "Provider",
    ):
        activity.add_column(name, overflow="crop")
    for row in data["recent"]:
        activity.add_row(
            safe(row["timestamp"], 19), safe(row["model"], 32), safe(row["status"], 20),
            str(row["preflight_input_tokens"]),
            f"{row['provider_input_tokens']} ["
            f"{safe(row.get('provider_measurement_label', 'UNKNOWN'), 20)}]",
            f"{row['candidate_context_tokens']} -> {row['compiled_context_tokens']}",
            f"{row.get('lexical_candidate_count', 0)}/"
            f"{row.get('dense_candidate_count', 0)}/{row.get('selected_memory_count', 0)}",
            safe(row.get("context_measurement_label", "UNKNOWN"), 20),
            str(row["context_tokens_avoided"]), str(row["graph_expanded_count"]),
            f"{row['retrieval_ms']:.1f} ms", f"{row['compilation_ms']:.1f} ms",
            f"{row.get('provider_ms', 0):.1f} ms",
        )
    if not data["recent"]:
        activity.add_row("No activity", *([""] * 12))
    sections: list[RenderableType] = [table, activity]
    graph = data.get("graph")
    if graph:
        graph_table = Table(title="Graph projection [MEASURED]")
        graph_table.add_column("Nodes")
        graph_table.add_column("Edges")
        graph_table.add_column("Supports")
        graph_table.add_column("State")
        graph_table.add_row(str(graph["nodes"]), str(graph["edges"]), str(graph["supports"]),
                            "dirty" if graph["dirty"] else "clean")
        sections.append(graph_table)
    breakdown = data.get("provider_models", [])
    if compare and breakdown:
        models_table = Table(title="Provider + model (successful invocations by context basis)")
        for name in (
            "Provider", "Model", "Runs", "Errors", "Candidate", "Compiled", "Avoided",
            "Reduction", "Basis",
        ):
            models_table.add_column(name, overflow="crop")
        for row in breakdown[:50]:
            ratio = row["weighted_reduction_ratio"]
            models_table.add_row(
                Text(safe(row["provider"])), Text(safe(row["model"])),
                str(row["invocations"]), str(row["errors"]),
                str(row["candidate_context_tokens"]), str(row["compiled_context_tokens"]),
                str(row["context_tokens_avoided"]),
                f"{ratio:.1%}" if ratio is not None else "UNKNOWN",
                Text(
                    safe(row["context_measurement_source"])
                    + " / "
                    + safe(row["context_tokenizer"])
                ),
            )
        sections.append(models_table)
    if (
        len(breakdown) == 1
        and breakdown[0]["invocations"] > 0
        and breakdown[0]["context_measurement_source"] != "unknown"
    ):
        row = breakdown[0]
        candidate = row["candidate_context_tokens"]
        compiled = row["compiled_context_tokens"]
        avoided = row["context_tokens_avoided"]
        maximum = max(candidate, compiled, avoided, 1)
        bars = Table(title="[MEASURED] Context tokens, one tokenizer basis")
        bars.add_column("Stage")
        bars.add_column("Scale")
        bars.add_column("Tokens", justify="right")
        for label, value in (
            ("Candidate", candidate), ("Compiled", compiled), ("Avoided", avoided)
        ):
            bars.add_row(label, "#" * round(24 * value / maximum), f"{value:,}")
        sections.append(bars)
    sections.append(
        Text(
            "Context and preflight counts use the recorded target tokenizer or approximation. "
            "Provider usage is separate; no cross-tokenizer totals are shown.",
            style="dim",
        )
    )
    return Group(*sections)
