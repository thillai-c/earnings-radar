"""
signal_aggregator.py — Combine deterministic and semantic layers into a TradingSignal.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import config
from models import (
    DeterministicMetrics,
    SemanticMetrics,
    SignalType,
    TradingSignal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

# Fields that count toward deterministic confidence
_DET_CONFIDENCE_FIELDS = [
    "current_revenue",
    "revenue_yoy_pct",
    "gross_margin_gaap",
    "eps_gaap_diluted",
    "free_cash_flow",
    "guided_revenue_midpoint",
    "operating_income",
    "net_income",
]


def _deterministic_confidence(det: DeterministicMetrics) -> float:
    """Return fraction of key deterministic fields that were successfully extracted."""
    populated = sum(
        1 for f in _DET_CONFIDENCE_FIELDS if getattr(det, f, None) is not None
    )
    return populated / len(_DET_CONFIDENCE_FIELDS)


def _overall_confidence(det: DeterministicMetrics, sem: SemanticMetrics) -> float:
    """Blend deterministic parse success rate with semantic confidence."""
    det_conf = _deterministic_confidence(det)
    sem_conf = sem.sentiment_confidence if sem.sentiment_confidence is not None else 0.5
    # Weight deterministic 40 %, semantic 60 %
    blended = 0.4 * det_conf + 0.6 * sem_conf
    return round(min(max(blended, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _classify_signal(score: float) -> SignalType:
    bands = config.SCORE_BANDS
    if score >= bands["STRONG_BUY"]:
        return SignalType.STRONG_BUY
    if score >= bands["BUY"]:
        return SignalType.BUY
    if score >= bands["WEAK_BUY"]:
        return SignalType.WEAK_BUY
    if score >= bands["HOLD"]:
        return SignalType.HOLD
    if score >= bands["WEAK_SELL"]:
        return SignalType.WEAK_SELL
    if score >= bands["SELL"]:
        return SignalType.SELL
    return SignalType.STRONG_SELL


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_deterministic(det: DeterministicMetrics) -> dict[str, float]:
    """Return a dict of contributor_name → point_value for the deterministic layer."""
    w = config.DET_WEIGHTS
    breakdown: dict[str, float] = {}

    yoy = det.revenue_yoy_pct

    if yoy is not None:
        if yoy > config.REVENUE_YOY_STRONG_THRESHOLD:
            breakdown["revenue_yoy_strong (>50%)"] = w["revenue_yoy_strong"]
        elif yoy > config.REVENUE_YOY_MODERATE_THRESHOLD:
            breakdown["revenue_yoy_moderate (>20%)"] = w["revenue_yoy_moderate"]
        elif yoy < config.REVENUE_YOY_NEGATIVE_THRESHOLD:
            breakdown["revenue_yoy_negative (<0%)"] = w["revenue_yoy_negative"]

    if det.margin_expansion is True:
        breakdown["margin_expansion"] = w["margin_expansion"]

    if det.guidance_raise is True:
        breakdown["guidance_raise"] = w["guidance_raise"]

    if det.free_cash_flow is not None and det.free_cash_flow > 0:
        breakdown["positive_fcf"] = w["positive_fcf"]

    if det.dividend_change_pct is not None and det.dividend_change_pct > 0:
        breakdown["dividend_increase"] = w["dividend_increase"]

    if det.buyback_authorized_usd is not None and det.buyback_authorized_usd > 0:
        breakdown["buyback_authorized"] = w["buyback_authorized"]

    if det.gross_margin_gaap is not None and det.gross_margin_gaap < config.GROSS_MARGIN_FLOOR:
        breakdown["low_gross_margin (<50%)"] = w["low_gross_margin"]

    return breakdown


def _score_semantic(sem: SemanticMetrics) -> dict[str, float]:
    """Return a dict of contributor_name → point_value for the semantic layer."""
    sw = config.SEM_WEIGHTS
    breakdown: dict[str, float] = {}
    conf = sem.sentiment_confidence if sem.sentiment_confidence is not None else 1.0

    if sem.sentiment:
        raw = sw["sentiment"].get(sem.sentiment, 0.0)
        weighted = round(raw * conf, 4)
        breakdown[f"sentiment ({sem.sentiment}, conf={conf:.0%})"] = weighted

    if sem.guidance_tone:
        pts = sw["guidance_tone"].get(sem.guidance_tone, 0.0)
        breakdown[f"guidance_tone ({sem.guidance_tone})"] = pts

    if sem.demand_signal:
        pts = sw["demand_signal"].get(sem.demand_signal, 0.0)
        breakdown[f"demand_signal ({sem.demand_signal})"] = pts

    if sem.management_tone:
        pts = sw["management_tone"].get(sem.management_tone, 0.0)
        breakdown[f"management_tone ({sem.management_tone})"] = pts

    if sem.beat_or_miss:
        pts = sw["beat_or_miss"].get(sem.beat_or_miss, 0.0)
        breakdown[f"beat_or_miss ({sem.beat_or_miss})"] = pts
        # Surprise magnitude bonus/penalty
        if sem.surprise_magnitude and sem.beat_or_miss in ("beat", "miss"):
            bonus = sw["surprise_magnitude_bonus"].get(sem.surprise_magnitude, 0.0)
            if sem.beat_or_miss == "miss":
                bonus = -bonus
            if bonus != 0.0:
                breakdown[f"surprise_magnitude ({sem.surprise_magnitude})"] = bonus

    return breakdown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    det: DeterministicMetrics,
    sem: SemanticMetrics,
) -> TradingSignal:
    """Combine *det* and *sem* into a :class:`TradingSignal`.

    Scores each contributor, sums to a final score in [-10, 10], classifies
    the signal, and computes an overall confidence estimate.

    Args:
        det: Metrics from the deterministic parser.
        sem: Metrics from the semantic parser.

    Returns:
        Fully populated :class:`TradingSignal`.
    """
    det_breakdown = _score_deterministic(det)
    sem_breakdown = _score_semantic(sem)

    combined = {**det_breakdown, **sem_breakdown}
    raw_score = sum(combined.values())
    score = round(max(min(raw_score, 10.0), -10.0), 4)

    signal_type = _classify_signal(score)
    confidence = _overall_confidence(det, sem)

    logger.info(
        "Signal aggregated: %s (score=%.2f, confidence=%.0f%%)",
        signal_type.value,
        score,
        confidence * 100,
    )

    return TradingSignal(
        signal=signal_type,
        score=score,
        confidence=confidence,
        deterministic_metrics=det,
        semantic_metrics=sem,
        score_breakdown=combined,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
