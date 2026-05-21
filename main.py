"""
main.py — CLI entrypoint for the trading signal system.

Usage:
    python main.py --input path/to/document.pdf [--output signal_output.json]
                   [--consensus-revenue 79.5] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-radar",
        description="Generate a structured trading signal from a financial document.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Path to the financial document (.pdf, .txt, or .html).",
    )
    parser.add_argument(
        "--output", "-o",
        default="signal_output.json",
        metavar="FILE",
        help="Path for the JSON output file (default: signal_output.json).",
    )
    parser.add_argument(
        "--consensus-revenue",
        type=float,
        default=None,
        metavar="BILLIONS",
        help="Wall Street consensus revenue estimate in billions (optional).",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip the semantic (LLM) layer and use deterministic only.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # ── Step 1: Load document ────────────────────────────────────────────────
    from document_loader import load_document
    try:
        doc = load_document(args.input)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR loading document: {exc}", file=sys.stderr)
        return 1

    # ── Step 2: Deterministic parser ─────────────────────────────────────────
    from deterministic_parser import parse as det_parse
    det_metrics = det_parse(
        text=doc.raw_text,
        consensus_revenue=args.consensus_revenue,
    )

    # ── Step 3: Semantic parser ───────────────────────────────────────────────
    from semantic_parser import parse as sem_parse
    from models import SemanticMetrics

    if args.no_semantic:
        logger.info("Semantic layer skipped (--no-semantic flag).")
        sem_metrics = SemanticMetrics()
    else:
        sem_metrics = sem_parse(doc.raw_text)

    # ── Step 4: Aggregate ─────────────────────────────────────────────────────
    from signal_aggregator import aggregate
    signal = aggregate(det_metrics, sem_metrics)

    # ── Step 5: Render and write output ───────────────────────────────────────
    from output import render
    render(signal, output_path=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
