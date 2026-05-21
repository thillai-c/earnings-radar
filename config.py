"""
config.py — Central configuration for all thresholds, model settings, and score bands.
Adjust values here to tune signal sensitivity without touching business logic.
"""

import os
from pathlib import Path

# Load .env from the project root if present (no-op if python-dotenv is absent)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

OPENAI_MODEL: str = "gpt-4o"
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
MAX_TOKENS: int = 1500

# ---------------------------------------------------------------------------
# Deterministic scoring thresholds
# ---------------------------------------------------------------------------

REVENUE_YOY_STRONG_THRESHOLD: float = 50.0    # % — earns +2 pts
REVENUE_YOY_MODERATE_THRESHOLD: float = 20.0  # % — earns +1 pt
REVENUE_YOY_NEGATIVE_THRESHOLD: float = 0.0   # % — below earns -2 pts
GROSS_MARGIN_FLOOR: float = 50.0              # % — below earns -1 pt

# ---------------------------------------------------------------------------
# Signal score bands  (lower bound, inclusive)
# ---------------------------------------------------------------------------

SCORE_BANDS: dict[str, float] = {
    "STRONG_BUY":  6.0,
    "BUY":         3.0,
    "WEAK_BUY":    1.0,
    "HOLD":        0.0,
    "WEAK_SELL":  -2.0,
    "SELL":       -5.0,
    "STRONG_SELL": float("-inf"),
}

# ---------------------------------------------------------------------------
# Deterministic point weights  (can be overridden by callers)
# ---------------------------------------------------------------------------

DET_WEIGHTS: dict[str, float] = {
    "revenue_yoy_strong":    2.0,
    "revenue_yoy_moderate":  1.0,
    "margin_expansion":      1.0,
    "guidance_raise":        1.0,
    "positive_fcf":          1.0,
    "dividend_increase":     1.0,
    "buyback_authorized":    1.0,
    "revenue_yoy_negative": -2.0,
    "low_gross_margin":     -1.0,
}

# ---------------------------------------------------------------------------
# Semantic point weights
# ---------------------------------------------------------------------------

SEM_WEIGHTS: dict[str, dict[str, float]] = {
    "sentiment": {
        "bullish":  2.0,
        "neutral":  0.0,
        "bearish": -2.0,
    },
    "guidance_tone": {
        "raised":     1.0,
        "initiated":  1.0,
        "maintained": 0.0,
        "lowered":   -2.0,
        "withdrawn": -3.0,
    },
    "demand_signal": {
        "accelerating":  1.0,
        "stable":        0.0,
        "decelerating": -1.0,
        "unclear":       0.0,
    },
    "management_tone": {
        "confident":  1.0,
        "mixed":      0.0,
        "cautious":  -1.0,
        "defensive": -2.0,
    },
    "beat_or_miss": {
        "beat":    1.0,
        "in-line": 0.0,
        "miss":   -2.0,
        "unknown": 0.0,
    },
    "surprise_magnitude_bonus": {
        "large":    1.0,
        "moderate": 0.5,
        "small":    0.0,
        "none":     0.0,
        "unknown":  0.0,
    },
}
