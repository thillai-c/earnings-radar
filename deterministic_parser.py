"""
deterministic_parser.py — Rule-based metric extraction from raw financial document text.

All extraction uses regex and numeric parsing only — no LLM calls.
Every field defaults to None if a reliable match cannot be found.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import config
from models import DeterministicMetrics, SegmentMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level numeric helpers
# ---------------------------------------------------------------------------

def _to_float(raw: str) -> Optional[float]:
    """Convert a raw string like '$81.6B', '74.9%', '2.39', '(1.2)' to float.

    Parentheses denote negative values.  Suffixes B/M/K are stripped (caller
    handles scaling via :func:`_scale_suffix`).  Returns None on failure.
    """
    if not raw:
        return None
    s = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # Strip trailing suffix letters so bare numbers like "81,615" still parse
    s = s.rstrip("BbMmKk")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def _scale_suffix(value: float, suffix: str) -> float:
    """Apply magnitude multiplier from a word or single-letter suffix."""
    s = suffix.strip().upper()
    if s in ("B", "BILLION"):
        return value * 1_000_000_000
    if s in ("M", "MILLION"):
        return value * 1_000_000
    if s in ("K", "THOUSAND"):
        return value * 1_000
    return value


def _parse_millions_table(value_str: str) -> Optional[float]:
    """Parse a bare integer/decimal assumed to be in millions (table context).

    E.g. '81,615' → 8.1615e10  (81,615 * 1e6)
    """
    val = _to_float(value_str)
    return val * 1_000_000 if val is not None else None


def _safe_pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


# ---------------------------------------------------------------------------
# Revenue
# ---------------------------------------------------------------------------

# Primary prose patterns: "revenue of $81.6 billion" or "revenue $81,615 million"
_REVENUE_PROSE = [
    r"revenue[s]?\s+(?:of\s+|was\s+|were\s+|totaled?\s+|for[^\n]{0,30}?of\s+)?"
    r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)\b",
    r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)\s+in\s+(?:total\s+)?revenue",
    r"(?:total\s+)?revenue[s]?\s*[:\|]\s*\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
]

# Summary table pattern: "Revenue $81,615 $68,127 $44,062 20% 85%"
# The three dollar amounts are Q_current, Q_prior, Q_year-ago (all in millions)
_REVENUE_TABLE = re.compile(
    r"^Revenue\s+\$?([\d,]+)\s+\$?([\d,]+)\s+\$?([\d,]+)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_revenue_block(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (current, prior_quarter, prior_year) revenue in USD.

    Tries table format first (most accurate), then falls back to prose patterns.
    """
    # Try the GAAP summary table first — gives all three periods in one line
    m = _REVENUE_TABLE.search(text)
    if m:
        curr = _parse_millions_table(m.group(1))
        pq   = _parse_millions_table(m.group(2))
        py   = _parse_millions_table(m.group(3))
        if curr:
            logger.debug("Revenue from table: curr=%.0f pq=%.0f py=%.0f", curr, pq or 0, py or 0)
            return curr, pq, py

    # Prose fallback
    current = None
    for pat in _REVENUE_PROSE:
        m2 = re.search(pat, text[:8000], re.IGNORECASE)
        if m2:
            val = _to_float(m2.group(1))
            if val is not None:
                current = _scale_suffix(val, m2.group(2))
                logger.debug("Revenue from prose: %.0f", current)
                break

    return current, None, None


# ---------------------------------------------------------------------------
# Explicit growth rate mentions
# ---------------------------------------------------------------------------

def _parse_growth_rates(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract explicitly stated YoY and QoQ revenue growth percentages."""
    yoy = qoq = None

    yoy_pats = [
        r"revenue[s]?[^\n]{0,60}up\s+([\d.]+)\s*%\s*from\s+a\s+year\s+ago",
        r"revenue[s]?[^\n]{0,60}(?:increased?|grew?)\s+([\d.]+)\s*%[^\n]{0,30}year",
        r"([\d.]+)\s*%\s+(?:increase|growth)\s+(?:from\s+)?(?:the\s+)?year[- ]ago",
        r"revenue[^\n]{0,80}Y/Y\s+([\d.]+)%",
    ]
    for pat in yoy_pats:
        m = re.search(pat, text[:10000], re.IGNORECASE)
        if m:
            val = _to_float(m.group(1))
            if val is not None:
                yoy = val
                logger.debug("YoY growth explicit: %.1f%%", yoy)
                break

    qoq_pats = [
        r"revenue[s]?[^\n]{0,60}up\s+([\d.]+)\s*%\s*from\s+the\s+previous\s+quarter",
        r"revenue[s]?[^\n]{0,60}up\s+([\d.]+)\s*%\s*(?:sequentially|from[^\n]{0,20}quarter)",
        r"([\d.]+)\s*%\s+sequentially",
    ]
    for pat in qoq_pats:
        m = re.search(pat, text[:10000], re.IGNORECASE)
        if m:
            val = _to_float(m.group(1))
            if val is not None:
                qoq = val
                logger.debug("QoQ growth explicit: %.1f%%", qoq)
                break

    return yoy, qoq


# ---------------------------------------------------------------------------
# Gross margin  — handles both combined-sentence and individual-sentence formats
# ---------------------------------------------------------------------------

def _parse_gross_margin(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (gaap_gross_margin_pct, non_gaap_gross_margin_pct)."""
    gaap = non_gaap = None

    # Combined sentence: "GAAP and non-GAAP gross margins were 74.9% and 75.0%"
    combined = re.search(
        r"gaap\s+and\s+non[- ]gaap\s+gross\s+margin[s]?\s+were\s+([\d.]+)%\s+and\s+([\d.]+)%",
        text, re.IGNORECASE
    )
    if combined:
        gaap = _to_float(combined.group(1))
        non_gaap = _to_float(combined.group(2))
        return gaap, non_gaap

    # Individual non-GAAP line
    ng_pats = [
        r"non[- ]gaap\s+gross\s+margin[s]?\s*(?:of|was|were|:|\*)?\s*([\d.]+)\s*%",
        r"gross\s+margin[s]?\s*[\(,]\s*non[- ]gaap[\),]?\s*(?:of|was|were|:)?\s*([\d.]+)\s*%",
    ]
    for pat in ng_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            non_gaap = _to_float(m.group(1))
            break

    # Individual GAAP line
    gaap_pats = [
        r"gaap\s+gross\s+margin[s]?\s*(?:of|was|were|:|\*)?\s*([\d.]+)\s*%",
        r"gross\s+margin[s]?\s*[\(,]\s*gaap[\),]?\s*(?:of|was|were|:)?\s*([\d.]+)\s*%",
        r"^Gross\s+margin\s+([\d.]+)%",   # table row: "Gross margin 74.9%"
    ]
    for pat in gaap_pats:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            gaap = _to_float(m.group(1))
            break

    return gaap, non_gaap


# ---------------------------------------------------------------------------
# EPS  — handles combined-sentence and table formats
# ---------------------------------------------------------------------------

def _parse_eps(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (eps_gaap_diluted, eps_non_gaap_diluted)."""
    gaap = non_gaap = None

    # Combined sentence: "GAAP and non-GAAP earnings per diluted share were $2.39 and $1.87"
    combined = re.search(
        r"gaap\s+and\s+non[- ]gaap\s+earnings\s+per\s+diluted\s+share\s+were\s+"
        r"\$?([\d.]+)\s+and\s+\$?([\d.]+)",
        text, re.IGNORECASE
    )
    if combined:
        gaap = _to_float(combined.group(1))
        non_gaap = _to_float(combined.group(2))
        return gaap, non_gaap

    # Non-GAAP standalone
    ng_pats = [
        r"non[- ]gaap\s+(?:diluted\s+)?(?:eps|earnings\s+per\s+(?:diluted\s+)?share)"
        r"[^\n]*?\$?([\d.]+)",
        r"diluted\s+earnings\s+per\s+share[^\n]*?non[- ]gaap[^\n]*?\$?([\d.]+)",
    ]
    for pat in ng_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            non_gaap = _to_float(m.group(1))
            break

    # GAAP standalone
    gaap_pats = [
        r"gaap\s+(?:diluted\s+)?(?:eps|earnings\s+per\s+(?:diluted\s+)?share)"
        r"[^\n]*?\$?([\d.]+)",
        r"diluted\s+earnings\s+per\s+share[^\n]*?gaap[^\n]*?\$?([\d.]+)",
        r"(?:diluted\s+)?earnings\s+per\s+(?:diluted\s+)?share\s+\$?([\d.]+)",
    ]
    for pat in gaap_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            gaap = _to_float(m.group(1))
            break

    return gaap, non_gaap


# ---------------------------------------------------------------------------
# EPS history from table for YoY/QoQ
# ---------------------------------------------------------------------------

# Table row: "Diluted earnings per share $2.39 $1.76 $0.76 36% 214%"
_EPS_TABLE = re.compile(
    r"Diluted\s+earnings\s+per\s+share\s+\$?([\d.]+)\s+\$?([\d.]+)\s+\$?([\d.]+)"
    r"(?:\s+(\d+)%\s+(\d+)%)?",
    re.IGNORECASE,
)


def _parse_eps_history(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (eps_qoq_pct, eps_yoy_pct) from the GAAP summary table."""
    m = _EPS_TABLE.search(text)
    if m and m.group(4) and m.group(5):
        return _to_float(m.group(4)), _to_float(m.group(5))
    if m:
        curr = _to_float(m.group(1))
        pq   = _to_float(m.group(2))
        py   = _to_float(m.group(3))
        return _safe_pct_change(curr, pq), _safe_pct_change(curr, py)
    return None, None


# ---------------------------------------------------------------------------
# Income items — table and prose
# ---------------------------------------------------------------------------

# Table: "Operating income $53,536 $44,299 $21,638 21% 147%"
_OI_TABLE = re.compile(
    r"^Operating\s+income\s+\$?([\d,]+)\s+\$?([\d,]+)\s+\$?([\d,]+)",
    re.IGNORECASE | re.MULTILINE,
)
_NI_TABLE = re.compile(
    r"^Net\s+income\s+\$?([\d,]+)\s+\$?([\d,]+)\s+\$?([\d,]+)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_income_items(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (operating_income, net_income) in USD."""
    oi = ni = None

    m = _OI_TABLE.search(text)
    if m:
        oi = _parse_millions_table(m.group(1))

    if oi is None:
        # Prose fallback
        for pat in [
            r"(?:gaap\s+)?operating\s+income[^\n]*?\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
            r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)\s+(?:gaap\s+)?operating\s+income",
        ]:
            m2 = re.search(pat, text[:12000], re.IGNORECASE)
            if m2:
                val = _to_float(m2.group(1))
                if val:
                    oi = _scale_suffix(val, m2.group(2))
                break

    m3 = _NI_TABLE.search(text)
    if m3:
        ni = _parse_millions_table(m3.group(1))

    if ni is None:
        for pat in [
            r"(?:gaap\s+)?net\s+income[^\n]*?\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
            r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)\s+(?:gaap\s+)?net\s+income",
        ]:
            m4 = re.search(pat, text[:12000], re.IGNORECASE)
            if m4:
                val = _to_float(m4.group(1))
                if val:
                    ni = _scale_suffix(val, m4.group(2))
                break

    return oi, ni


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

def _parse_guidance(text: str) -> tuple[
    Optional[float], Optional[tuple[float, float]],
    Optional[float], Optional[float]
]:
    """Return (midpoint_usd, (lo_usd, hi_usd), gaap_gm_pct, non_gaap_gm_pct)."""
    midpoint = lo = hi = gaap_gm = non_gaap_gm = None

    # "$91.0 billion, plus or minus 2%"
    # Non-greedy [^\n]{0,120}? prevents consuming the dollar amount itself.
    range_pct = re.search(
        r"(?:revenue|outlook)[^\n]{0,120}?"
        r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)"
        r"[^\n]{0,50}(?:plus\s+or\s+minus|±|\+/?-)\s*([\d.]+)\s*%",
        text, re.IGNORECASE,
    )
    if range_pct:
        mid_val = _to_float(range_pct.group(1))
        if mid_val is not None:
            mid_val = _scale_suffix(mid_val, range_pct.group(2))
            tol_pct = _to_float(range_pct.group(3))
            if tol_pct is not None:
                tol = mid_val * tol_pct / 100
                lo, hi = mid_val - tol, mid_val + tol
            midpoint = mid_val
            logger.debug("Guidance midpoint: %.2f", midpoint)

    # "$X to $Y billion" range
    if midpoint is None:
        range_abs = re.search(
            r"(?:revenue|outlook)[^\n]{0,80}"
            r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)"
            r"[^\n]{0,10}to[^\n]{0,10}"
            r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
            text, re.IGNORECASE,
        )
        if range_abs:
            v1 = _to_float(range_abs.group(1))
            v2 = _to_float(range_abs.group(3))
            if v1 and v2:
                lo = _scale_suffix(v1, range_abs.group(2))
                hi = _scale_suffix(v2, range_abs.group(4))
                midpoint = (lo + hi) / 2

    # Guidance gross margins: "GAAP and non-GAAP gross margins are expected to be 74.9% and 75.0%"
    combined_gm = re.search(
        r"gaap\s+and\s+non[- ]gaap\s+gross\s+margin[s]?\s+are\s+expected\s+to\s+be\s+"
        r"([\d.]+)%\s+and\s+([\d.]+)%",
        text, re.IGNORECASE,
    )
    if combined_gm:
        gaap_gm = _to_float(combined_gm.group(1))
        non_gaap_gm = _to_float(combined_gm.group(2))
    else:
        # Individual patterns
        ng_gm = re.search(
            r"(?:guidance|outlook|expected)[^\n]{0,120}"
            r"non[- ]gaap\s+gross\s+margin[^\n]{0,30}([\d.]+)\s*%",
            text, re.IGNORECASE,
        )
        if ng_gm:
            non_gaap_gm = _to_float(ng_gm.group(1))

        g_gm = re.search(
            r"(?:guidance|outlook|expected)[^\n]{0,120}"
            r"gaap\s+gross\s+margin[^\n]{0,30}([\d.]+)\s*%",
            text, re.IGNORECASE,
        )
        if g_gm:
            gaap_gm = _to_float(g_gm.group(1))

    guided_range = (lo, hi) if lo is not None and hi is not None else None
    return midpoint, guided_range, gaap_gm, non_gaap_gm


# ---------------------------------------------------------------------------
# Shareholder returns
# ---------------------------------------------------------------------------

def _parse_shareholder_returns(text: str) -> tuple[
    Optional[float], Optional[float], Optional[float]
]:
    """Return (buyback_usd, dividend_new, dividend_old)."""
    buyback = div_new = div_old = None

    # Repurchase authorization
    bb_pats = [
        # "approved an additional $80.0 billion to the Company's share repurchase authorization"
        r"(?:authorized?|approved)[^\n]{0,80}\$?([\d,]+\.?\d*)\s*(billion|million|B|M)"
        r"[^\n]{0,80}(?:share\s+repurchase|buyback|repurchase\s+authorization)",
        r"(?:share\s+repurchase|buyback|repurchase\s+authorization)[^\n]{0,80}"
        r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
        # "$X billion additional share repurchase"
        r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)\s+additional\s+share\s+repurchase",
        r"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)[^\n]{0,30}repurchase\s+authorization",
    ]
    for pat in bb_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _to_float(m.group(1))
            if val:
                buyback = _scale_suffix(val, m.group(2))
                logger.debug("Buyback authorized: %.2f", buyback)
                break

    # Dividend — new and old
    # "increasing its quarterly cash dividend from $0.01 per share to $0.25 per share"
    div_change = re.search(
        r"(?:increasing|raising|changing)[^\n]{0,50}dividend\s+from\s+\$?([\d.]+)\s*per\s+share"
        r"[^\n]{0,30}to\s+\$?([\d.]+)\s*per\s+share",
        text, re.IGNORECASE,
    )
    if div_change:
        div_old = _to_float(div_change.group(1))
        div_new = _to_float(div_change.group(2))
    else:
        # Single mention of new dividend
        div_pat = re.search(
            r"(?:quarterly\s+)?(?:cash\s+)?dividend[^\n]{0,60}\$?([\d.]+)\s*per\s+share",
            text, re.IGNORECASE,
        )
        if div_pat:
            div_new = _to_float(div_pat.group(1))

    return buyback, div_new, div_old


# ---------------------------------------------------------------------------
# Cash flow  — table and prose
# ---------------------------------------------------------------------------

# Table: "Free cash flow $ 48,554 $ 34,902 $ 26,135"
_FCF_TABLE = re.compile(
    r"Free\s+cash\s+flow\s+\$?\s*([\d,]+)\s+\$?\s*([\d,]+)",
    re.IGNORECASE,
)
_OCF_TABLE = re.compile(
    r"(?:cash\s+flows?\s+from\s+operating\s+activities?|net\s+cash\s+provided\s+by\s+operating)"
    r"\s+\$?\s*([\d,]+)",
    re.IGNORECASE,
)


def _parse_cash_flow(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (operating_cash_flow, free_cash_flow, capex) in USD."""
    ocf = fcf = capex = None

    m_fcf = _FCF_TABLE.search(text)
    if m_fcf:
        fcf = _parse_millions_table(m_fcf.group(1))
        logger.debug("FCF from table: %.0f", fcf or 0)

    if fcf is None:
        m2 = re.search(
            r"free\s+cash\s+flow[^\n]{0,60}\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
            text, re.IGNORECASE,
        )
        if m2:
            val = _to_float(m2.group(1))
            if val:
                fcf = _scale_suffix(val, m2.group(2))

    m_ocf = _OCF_TABLE.search(text)
    if m_ocf:
        ocf = _parse_millions_table(m_ocf.group(1))

    if ocf is None:
        m3 = re.search(
            r"(?:cash\s+flow[s]?\s+from\s+operations?|operating\s+cash\s+flow[s]?)"
            r"[^\n]{0,60}\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
            text, re.IGNORECASE,
        )
        if m3:
            val = _to_float(m3.group(1))
            if val:
                ocf = _scale_suffix(val, m3.group(2))

    capex_m = re.search(
        r"capital\s+expenditures?[^\n]{0,60}\$?([\d,]+\.?\d*)\s*(billion|million|B|M)",
        text, re.IGNORECASE,
    )
    if capex_m:
        val = _to_float(capex_m.group(1))
        if val:
            capex = _scale_suffix(val, capex_m.group(2))

    return ocf, fcf, capex


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------

_KNOWN_SEGMENTS = [
    "data center", "edge computing", "gaming", "professional visualization",
    "automotive", "oem", "compute & networking", "graphics", "cloud",
    "enterprise", "consumer", "intelligent cloud", "productivity",
    "personal computing", "services", "products", "server", "pc client",
]


def _parse_segments(text: str) -> list[SegmentMetrics]:
    """Extract per-segment revenue figures."""
    segments: list[SegmentMetrics] = []
    seen: set[str] = set()

    for seg_name in _KNOWN_SEGMENTS:
        # Non-greedy [^\n]{0,60}? to avoid consuming the dollar amount.
        pat = (
            rf"(?:{re.escape(seg_name)})[^\n]{{0,60}}?"
            rf"(?:revenue[s]?\s+(?:of\s+|was\s+|were\s+)?)?"
            rf"\$?([\d,]+\.?\d*)\s*(billion|million|B|M)"
        )
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _to_float(m.group(1))
            if val is None:
                continue
            revenue = _scale_suffix(val, m.group(2))
            key = seg_name.lower()
            if key not in seen:
                seen.add(key)
                segments.append(SegmentMetrics(name=seg_name.title(), revenue=revenue))
                logger.debug("Segment '%s' revenue: %.2f", seg_name, revenue)

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(
    text: str,
    consensus_revenue: Optional[float] = None,
    prior_year_gross_margin: Optional[float] = None,
) -> DeterministicMetrics:
    """Extract structured metrics from *text* using deterministic rules.

    Args:
        text: Raw text of the financial document.
        consensus_revenue: Optional Wall Street consensus revenue estimate (USD billions).
        prior_year_gross_margin: Known prior-year gross margin for margin expansion check.

    Returns:
        :class:`DeterministicMetrics` with all extractable fields populated.
        Fields that cannot be extracted are set to None.
    """
    metrics = DeterministicMetrics()

    # Revenue
    try:
        curr, pq, py = _parse_revenue_block(text)
        metrics.current_revenue = curr
        metrics.prior_quarter_revenue = pq
        metrics.prior_year_revenue = py
    except Exception as exc:
        logger.warning("Revenue block parse error: %s", exc)

    # Growth rates — prefer explicit mentions; derive from absolutes as fallback
    try:
        yoy_expl, qoq_expl = _parse_growth_rates(text)
        metrics.revenue_yoy_pct = yoy_expl or _safe_pct_change(
            metrics.current_revenue, metrics.prior_year_revenue
        )
        metrics.revenue_qoq_pct = qoq_expl or _safe_pct_change(
            metrics.current_revenue, metrics.prior_quarter_revenue
        )
    except Exception as exc:
        logger.warning("Growth rate parse error: %s", exc)

    # Gross margin
    try:
        metrics.gross_margin_gaap, metrics.gross_margin_non_gaap = _parse_gross_margin(text)
    except Exception as exc:
        logger.warning("Gross margin parse error: %s", exc)

    # EPS (current quarter)
    try:
        metrics.eps_gaap_diluted, metrics.eps_non_gaap_diluted = _parse_eps(text)
    except Exception as exc:
        logger.warning("EPS parse error: %s", exc)

    # EPS history / growth
    try:
        metrics.eps_qoq_pct, metrics.eps_yoy_pct = _parse_eps_history(text)
    except Exception as exc:
        logger.warning("EPS history parse error: %s", exc)

    # Income items
    try:
        metrics.operating_income, metrics.net_income = _parse_income_items(text)
    except Exception as exc:
        logger.warning("Income parse error: %s", exc)

    # Guidance
    try:
        (
            metrics.guided_revenue_midpoint,
            metrics.guided_revenue_range,
            metrics.guided_gross_margin_gaap,
            metrics.guided_gross_margin_non_gaap,
        ) = _parse_guidance(text)
    except Exception as exc:
        logger.warning("Guidance parse error: %s", exc)

    # Shareholder returns
    try:
        (
            metrics.buyback_authorized_usd,
            metrics.dividend_per_share_new,
            metrics.dividend_per_share_old,
        ) = _parse_shareholder_returns(text)
        if metrics.dividend_per_share_new and metrics.dividend_per_share_old:
            metrics.dividend_change_pct = _safe_pct_change(
                metrics.dividend_per_share_new, metrics.dividend_per_share_old
            )
    except Exception as exc:
        logger.warning("Shareholder return parse error: %s", exc)

    # Cash flow
    try:
        (
            metrics.operating_cash_flow,
            metrics.free_cash_flow,
            metrics.capex,
        ) = _parse_cash_flow(text)
    except Exception as exc:
        logger.warning("Cash flow parse error: %s", exc)

    # Segments
    try:
        metrics.segments = _parse_segments(text)
    except Exception as exc:
        logger.warning("Segment parse error: %s", exc)

    # Guidance vs consensus
    if consensus_revenue is not None and metrics.guided_revenue_midpoint is not None:
        metrics.guidance_vs_consensus_pct = _safe_pct_change(
            metrics.guided_revenue_midpoint, consensus_revenue * 1e9
        )

    # Beat / miss flags
    if metrics.revenue_yoy_pct is not None:
        metrics.revenue_beat = metrics.revenue_yoy_pct > config.REVENUE_YOY_MODERATE_THRESHOLD

    if metrics.gross_margin_gaap is not None and prior_year_gross_margin is not None:
        metrics.margin_expansion = metrics.gross_margin_gaap > prior_year_gross_margin
    elif metrics.gross_margin_gaap is not None and metrics.prior_year_revenue is not None:
        # Use extracted prior-year margin from GAAP table if available
        py_gm = _parse_prior_year_gross_margin(text)
        if py_gm is not None:
            metrics.margin_expansion = metrics.gross_margin_gaap > py_gm

    if metrics.guided_revenue_midpoint is not None and metrics.current_revenue is not None:
        metrics.guidance_raise = metrics.guided_revenue_midpoint > metrics.current_revenue

    return metrics


def _parse_prior_year_gross_margin(text: str) -> Optional[float]:
    """Extract prior-year gross margin from summary table for margin expansion check.

    Looks for "Gross margin  74.9%  75.0%  60.5%" style rows.
    """
    m = re.search(
        r"^Gross\s+margin\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%",
        text, re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return _to_float(m.group(3))  # third column = Q1 prior year
    return None
