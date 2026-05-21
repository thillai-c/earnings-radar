"""
models.py — All dataclasses and enums for the trading signal system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class DocumentType(str, Enum):
    EARNINGS_RELEASE = "earnings_release"
    PRESS_RELEASE = "press_release"
    TEN_K = "10k"
    TEN_Q = "10q"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Document input
# ---------------------------------------------------------------------------

@dataclass
class DocumentInput:
    """Represents a loaded financial document ready for parsing."""

    raw_text: str
    source_file: str
    detected_type: DocumentType
    page_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Segment-level metrics
# ---------------------------------------------------------------------------

@dataclass
class SegmentMetrics:
    """Revenue and growth metrics for a single business segment."""

    name: str
    revenue: Optional[float] = None          # USD billions
    qoq_pct: Optional[float] = None
    yoy_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Deterministic layer output
# ---------------------------------------------------------------------------

@dataclass
class DeterministicMetrics:
    """Structured metrics extracted deterministically via regex / numeric parsing."""

    # Revenue
    current_revenue: Optional[float] = None
    prior_quarter_revenue: Optional[float] = None
    prior_year_revenue: Optional[float] = None
    revenue_qoq_pct: Optional[float] = None
    revenue_yoy_pct: Optional[float] = None

    # Profitability
    gross_margin_gaap: Optional[float] = None
    gross_margin_non_gaap: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_gaap_diluted: Optional[float] = None
    eps_non_gaap_diluted: Optional[float] = None
    eps_qoq_pct: Optional[float] = None
    eps_yoy_pct: Optional[float] = None

    # Guidance
    guided_revenue_midpoint: Optional[float] = None
    guided_revenue_range: Optional[tuple[float, float]] = None
    guided_gross_margin_gaap: Optional[float] = None
    guided_gross_margin_non_gaap: Optional[float] = None
    guidance_vs_consensus_pct: Optional[float] = None

    # Shareholder returns
    buyback_authorized_usd: Optional[float] = None
    dividend_per_share_new: Optional[float] = None
    dividend_per_share_old: Optional[float] = None
    dividend_change_pct: Optional[float] = None

    # Cash flow
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # Segments
    segments: list[SegmentMetrics] = field(default_factory=list)

    # Beat / miss flags
    revenue_beat: Optional[bool] = None
    margin_expansion: Optional[bool] = None
    guidance_raise: Optional[bool] = None


# ---------------------------------------------------------------------------
# Semantic layer output
# ---------------------------------------------------------------------------

@dataclass
class SemanticMetrics:
    """Qualitative signals extracted by the LLM."""

    sentiment: Optional[str] = None                  # bullish | bearish | neutral
    sentiment_confidence: Optional[float] = None     # 0.0 – 1.0
    guidance_tone: Optional[str] = None              # raised | lowered | maintained | initiated | withdrawn
    management_tone: Optional[str] = None            # confident | cautious | mixed | defensive
    key_risks: list[str] = field(default_factory=list)
    key_catalysts: list[str] = field(default_factory=list)
    demand_signal: Optional[str] = None              # accelerating | stable | decelerating | unclear
    competitive_signal: Optional[str] = None         # gaining_share | losing_share | stable | unclear
    capital_allocation_signal: Optional[str] = None  # shareholder_friendly | reinvesting | mixed
    beat_or_miss: Optional[str] = None               # beat | miss | in-line | unknown
    surprise_magnitude: Optional[str] = None         # large | moderate | small | none | unknown
    sector_read_through: Optional[str] = None
    headline_summary: Optional[str] = None
    notable_quotes: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Final trading signal
# ---------------------------------------------------------------------------

@dataclass
class TradingSignal:
    """Aggregated trading signal combining deterministic and semantic layers."""

    signal: SignalType
    score: float                              # -10 to +10
    confidence: float                         # 0 to 1
    deterministic_metrics: DeterministicMetrics
    semantic_metrics: SemanticMetrics
    score_breakdown: dict[str, float] = field(default_factory=dict)
    generated_at: str = ""                    # ISO 8601 timestamp
