"""
semantic_parser.py — LLM-based qualitative signal extraction via the OpenAI API.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import config
from models import SemanticMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior equity analyst specializing in technology companies. \
You will be given a financial document (earnings release, 10-K excerpt, press release, etc.). \
Extract the following signals and return them as a single JSON object with no preamble or \
markdown. All fields are required; use null if you cannot determine a value.

JSON schema:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_confidence": 0.0-1.0,
  "guidance_tone": "raised" | "lowered" | "maintained" | "initiated" | "withdrawn",
  "management_tone": "confident" | "cautious" | "mixed" | "defensive",
  "key_risks": ["string", ...],
  "key_catalysts": ["string", ...],
  "demand_signal": "accelerating" | "stable" | "decelerating" | "unclear",
  "competitive_signal": "gaining_share" | "losing_share" | "stable" | "unclear",
  "capital_allocation_signal": "shareholder_friendly" | "reinvesting" | "mixed",
  "beat_or_miss": "beat" | "miss" | "in-line" | "unknown",
  "surprise_magnitude": "large" | "moderate" | "small" | "none" | "unknown",
  "sector_read_through": "string",
  "headline_summary": "string",
  "notable_quotes": ["string", ...],
  "red_flags": ["string", ...]
}

Rules:
- key_risks: up to 5, ranked by severity
- key_catalysts: up to 5, ranked by magnitude
- notable_quotes: up to 3 verbatim quotes from management
- red_flags: anything unusual — guidance exclusions, one-time items, accounting changes
- headline_summary: 2 sentences maximum
- sector_read_through: brief implication for the broader sector"""

RETRY_REMINDER = (
    "Return ONLY raw JSON — no markdown, no prose, no code fences. "
    "Start your response with '{'."
)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """Extract the JSON object from *raw*, handling common LLM formatting slips."""
    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    # Find first '{' to last '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start:end])


# ---------------------------------------------------------------------------
# Dataclass builder
# ---------------------------------------------------------------------------

def _build_semantic_metrics(data: dict) -> SemanticMetrics:
    """Populate a SemanticMetrics dataclass from the parsed LLM JSON dict."""

    def _str_or_none(key: str) -> Optional[str]:
        val = data.get(key)
        return str(val).lower() if val is not None else None

    def _float_or_none(key: str) -> Optional[float]:
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _list_of_str(key: str) -> list[str]:
        val = data.get(key)
        if isinstance(val, list):
            return [str(v) for v in val if v]
        return []

    return SemanticMetrics(
        sentiment=_str_or_none("sentiment"),
        sentiment_confidence=_float_or_none("sentiment_confidence"),
        guidance_tone=_str_or_none("guidance_tone"),
        management_tone=_str_or_none("management_tone"),
        key_risks=_list_of_str("key_risks"),
        key_catalysts=_list_of_str("key_catalysts"),
        demand_signal=_str_or_none("demand_signal"),
        competitive_signal=_str_or_none("competitive_signal"),
        capital_allocation_signal=_str_or_none("capital_allocation_signal"),
        beat_or_miss=_str_or_none("beat_or_miss"),
        surprise_magnitude=_str_or_none("surprise_magnitude"),
        sector_read_through=data.get("sector_read_through"),
        headline_summary=data.get("headline_summary"),
        notable_quotes=_list_of_str("notable_quotes"),
        red_flags=_list_of_str("red_flags"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(document_text: str) -> SemanticMetrics:
    """Send *document_text* to the OpenAI API and return :class:`SemanticMetrics`.

    Retries once with an explicit JSON-only reminder if the first response cannot
    be parsed.  Falls back to an empty :class:`SemanticMetrics` on any failure so
    the deterministic layer can still produce a signal.

    Args:
        document_text: Raw text of the financial document.

    Returns:
        :class:`SemanticMetrics` populated from the LLM response.
    """
    if not config.OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY not set — skipping semantic analysis."
        )
        return SemanticMetrics()

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.error("openai package not installed — skipping semantic layer.")
        return SemanticMetrics()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _call(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    # First attempt
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": document_text},
    ]
    try:
        raw = _call(messages)
        logger.debug("Semantic parser raw response (first attempt, %d chars)", len(raw))
        data = _parse_json_response(raw)
        return _build_semantic_metrics(data)
    except Exception as exc:
        logger.warning("First semantic parse attempt failed (%s); retrying.", exc)

    # Retry with JSON-only reminder
    try:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": RETRY_REMINDER})
        raw2 = _call(messages)
        logger.debug("Semantic parser raw response (retry, %d chars)", len(raw2))
        data2 = _parse_json_response(raw2)
        return _build_semantic_metrics(data2)
    except Exception as exc:
        logger.error(
            "Semantic parser failed after retry (%s). Returning empty SemanticMetrics.", exc
        )
        return SemanticMetrics()
