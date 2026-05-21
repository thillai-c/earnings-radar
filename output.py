"""
output.py — Rich terminal output and JSON serialization for TradingSignal.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.columns import Columns

from models import SignalType, TradingSignal

logger = logging.getLogger(__name__)

# Force VT100 rendering on Windows to avoid cp1252 encoding errors with
# box-drawing characters.  Legacy Windows Console API renderer is bypassed.
console = Console(legacy_windows=False)

# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

_SIGNAL_STYLES: dict[SignalType, str] = {
    SignalType.STRONG_BUY:  "bold bright_green",
    SignalType.BUY:         "green",
    SignalType.WEAK_BUY:    "dark_green",
    SignalType.HOLD:        "yellow",
    SignalType.WEAK_SELL:   "dark_orange",
    SignalType.SELL:        "red",
    SignalType.STRONG_SELL: "bold bright_red",
}


def _signal_label(sig: SignalType) -> Text:
    style = _SIGNAL_STYLES.get(sig, "white")
    return Text(sig.value.replace("_", " "), style=style)


def _fmt_pct(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def _fmt_usd(val: Optional[float], unit: str = "B") -> str:
    if val is None:
        return "N/A"
    if unit == "B":
        return f"${val / 1e9:.1f}B"
    if unit == "M":
        return f"${val / 1e6:.1f}M"
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header_panel(signal: TradingSignal) -> Panel:
    sig_label = _signal_label(signal.signal)
    score_str = f"score: {signal.score:+.1f} / 10"
    conf_str = f"Confidence: {signal.confidence:.0%}"
    ts_str = f"Generated:  {signal.generated_at}"

    content = Text()
    content.append("Signal:      ")
    content.append_text(sig_label)
    content.append(f"  ({score_str})\n")
    content.append(f"{conf_str}\n")
    content.append(ts_str)

    return Panel(content, title="[bold cyan]TRADING SIGNAL REPORT[/bold cyan]", box=box.DOUBLE)


def _headline_section(signal: TradingSignal) -> str:
    summary = signal.semantic_metrics.headline_summary or "No summary available."
    return f"[bold]HEADLINE[/bold]\n{'─' * 40}\n{summary}"


def _key_metrics_section(signal: TradingSignal) -> Table:
    det = signal.deterministic_metrics
    table = Table(
        title="KEY METRICS",
        box=box.SIMPLE_HEAD,
        show_header=False,
        padding=(0, 1),
        title_style="bold",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value")

    rev = _fmt_usd(det.current_revenue)
    yoy = _fmt_pct(det.revenue_yoy_pct)
    qoq = _fmt_pct(det.revenue_qoq_pct)
    table.add_row("Revenue", f"{rev}  ({yoy} YoY, {qoq} QoQ)")

    gm_gaap = f"{det.gross_margin_gaap:.1f}%" if det.gross_margin_gaap is not None else "N/A"
    gm_ng = f"{det.gross_margin_non_gaap:.1f}%" if det.gross_margin_non_gaap is not None else "N/A"
    table.add_row("Gross Margin", f"{gm_gaap} GAAP / {gm_ng} Non-GAAP")

    eps_g = f"${det.eps_gaap_diluted:.2f}" if det.eps_gaap_diluted is not None else "N/A"
    eps_ng = f"${det.eps_non_gaap_diluted:.2f}" if det.eps_non_gaap_diluted is not None else "N/A"
    eps_yoy = _fmt_pct(det.eps_yoy_pct)
    table.add_row("EPS (diluted)", f"{eps_g} GAAP / {eps_ng} Non-GAAP  ({eps_yoy} YoY)")

    table.add_row("Free Cash Flow", _fmt_usd(det.free_cash_flow))
    table.add_row("Operating Income", _fmt_usd(det.operating_income))

    if det.guided_revenue_midpoint is not None:
        gr = _fmt_usd(det.guided_revenue_midpoint)
        if det.guided_revenue_range and None not in det.guided_revenue_range:
            lo, hi = det.guided_revenue_range
            gr += f" (range {_fmt_usd(lo)} – {_fmt_usd(hi)})"
        table.add_row("Next-Quarter Guidance", gr)

    if det.buyback_authorized_usd:
        table.add_row("Buyback Authorized", _fmt_usd(det.buyback_authorized_usd))

    return table


def _score_breakdown_table(signal: TradingSignal) -> Table:
    table = Table(
        title="SCORE BREAKDOWN",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        show_footer=True,
    )
    table.add_column("Contributor", style="cyan")
    table.add_column("Points", justify="right", footer=f"{signal.score:+.2f}")

    sorted_items = sorted(signal.score_breakdown.items(), key=lambda x: -abs(x[1]))
    for name, pts in sorted_items:
        color = "green" if pts > 0 else ("red" if pts < 0 else "white")
        table.add_row(name, Text(f"{pts:+.2f}", style=color))

    return table


def _semantic_section(signal: TradingSignal) -> Table:
    sem = signal.semantic_metrics
    table = Table(
        title="SEMANTIC ANALYSIS",
        box=box.SIMPLE_HEAD,
        show_header=False,
        title_style="bold",
    )
    table.add_column("Label", style="cyan", no_wrap=True)
    table.add_column("Value")

    conf = f" (confidence: {sem.sentiment_confidence:.0%})" if sem.sentiment_confidence else ""
    table.add_row("Sentiment", f"{(sem.sentiment or 'N/A').title()}{conf}")
    table.add_row("Guidance Tone", (sem.guidance_tone or "N/A").title())
    table.add_row("Demand Signal", (sem.demand_signal or "N/A").title())
    table.add_row("Mgmt Tone", (sem.management_tone or "N/A").title())
    table.add_row("Competitive Signal", (sem.competitive_signal or "N/A").replace("_", " ").title())
    table.add_row("Capital Allocation", (sem.capital_allocation_signal or "N/A").replace("_", " ").title())

    return table


def _numbered_list(items: list[str], fallback: str = "None identified.") -> str:
    if not items:
        return fallback
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(signal: TradingSignal, output_path: Optional[str] = None) -> None:
    """Render the trading signal report to the terminal and optionally write JSON.

    Args:
        signal: The :class:`TradingSignal` to display.
        output_path: If provided, write the full signal as JSON to this path.
    """
    console.print()
    console.print(_header_panel(signal))
    console.print()

    # Headline
    console.print(_headline_section(signal))
    console.print()

    # Key metrics table
    console.print(_key_metrics_section(signal))
    console.print()

    # Score breakdown
    console.print(_score_breakdown_table(signal))
    console.print()

    # Semantic analysis
    console.print(_semantic_section(signal))
    console.print()

    sem = signal.semantic_metrics

    # Key catalysts
    console.rule("[bold]KEY CATALYSTS[/bold]")
    console.print(_numbered_list(sem.key_catalysts))
    console.print()

    # Key risks
    console.rule("[bold]KEY RISKS[/bold]")
    console.print(_numbered_list(sem.key_risks))
    console.print()

    # Red flags
    console.rule("[bold]RED FLAGS[/bold]")
    console.print(_numbered_list(sem.red_flags))
    console.print()

    # Notable quotes
    if sem.notable_quotes:
        console.rule("[bold]NOTABLE QUOTES[/bold]")
        for q in sem.notable_quotes:
            console.print(f'  "[italic]{q}[/italic]"')
        console.print()

    # Sector read-through
    if sem.sector_read_through:
        console.rule("[bold]SECTOR READ-THROUGH[/bold]")
        console.print(sem.sector_read_through)
        console.print()

    # Segment breakdown
    if signal.deterministic_metrics.segments:
        table = Table(title="SEGMENT REVENUE", box=box.SIMPLE_HEAD, title_style="bold")
        table.add_column("Segment", style="cyan")
        table.add_column("Revenue", justify="right")
        table.add_column("QoQ", justify="right")
        table.add_column("YoY", justify="right")
        for seg in signal.deterministic_metrics.segments:
            table.add_row(
                seg.name,
                _fmt_usd(seg.revenue),
                _fmt_pct(seg.qoq_pct),
                _fmt_pct(seg.yoy_pct),
            )
        console.print(table)
        console.print()

    # JSON output
    if output_path:
        _write_json(signal, output_path)
        console.print(f"[dim]Full signal written to: {output_path}[/dim]")


def _signal_to_dict(signal: TradingSignal) -> dict:
    """Convert TradingSignal to a JSON-serialisable dict."""

    def _serialise(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _serialise(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(i) for i in obj]
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return obj

    return _serialise(signal)


def _write_json(signal: TradingSignal, path: str) -> None:
    """Serialise *signal* to JSON and write to *path*."""
    try:
        data = _signal_to_dict(signal)
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Signal JSON written to '%s'", path)
    except Exception as exc:
        logger.error("Failed to write JSON output to '%s': %s", path, exc)
