"""
Build retrieval indexes for every 10-K already cached by the EDGAR
module.

Inputs : data/EDGAR_retrieval/cache/<TICKER>/<accession>.txt + .meta.json
Outputs: data/rag_10k/cache/<TICKER>/<accession>.chunks.json
                                     /<accession>.embeddings.npz

Per-ticker failures do not stop the run; a summary is printed at the
end and the script exits non-zero if anything failed. Cached indexes
are returned without re-embedding, so subsequent runs are near-instant.

Usage from data/rag_10k/:
    python run_build_indexes.py
"""

import json
import sys
import time
from pathlib import Path

from rag_10k import DEFAULT_MODEL, build_or_load_index


EDGAR_CACHE = Path(__file__).parent.parent / "EDGAR_retrieval" / "cache"
RAG_CACHE = Path(__file__).parent / "cache"


def _discover_filings():
    """
    Yield (ticker, accession_number, text_path) for every cached
    EDGAR filing.
    """
    for ticker_dir in sorted(EDGAR_CACHE.iterdir()):
        if not ticker_dir.is_dir():
            continue

        for meta_path in sorted(ticker_dir.glob("*.meta.json")):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            text_path = ticker_dir / f"{meta['accession_number']}.txt"

            if not text_path.exists():
                continue

            yield meta["ticker"], meta["accession_number"], text_path


def main():
    RAG_CACHE.mkdir(parents=True, exist_ok=True)

    filings = list(_discover_filings())

    if not filings:
        print(f"No EDGAR cache found at {EDGAR_CACHE}.")
        print("Run data/EDGAR_retrieval/run_fetch.py first.")
        sys.exit(1)

    print(f"Found {len(filings)} cached filings. Model = {DEFAULT_MODEL}.")
    print("First run downloads model weights (~130 MB) once.")
    print()

    successes = []
    failures = []

    started = time.monotonic()

    for i, (ticker, accession, text_path) in enumerate(filings, start=1):
        print(f"[{i:2d}/{len(filings)}] {ticker} {accession}", flush=True)

        try:
            text = text_path.read_text(encoding="utf-8")

            t_start = time.monotonic()
            index = build_or_load_index(
                ticker, accession, text,
                cache_dir=RAG_CACHE,
            )
            t_elapsed = time.monotonic() - t_start

            sections = sorted({c["section_code"] for c in index["chunks"]})

            print(
                f"           -> chunks={len(index['chunks']):>4d}  "
                f"sections={len(sections):>2d}  "
                f"{t_elapsed:>5.1f}s"
            )

            successes.append((ticker, accession, index))

        except Exception as exc:
            print(f"           -> FAILED: {type(exc).__name__}: {exc}")
            failures.append((ticker, accession, exc))

    elapsed = time.monotonic() - started

    print()
    print(
        f"Done in {elapsed:.1f}s: "
        f"{len(successes)} succeeded, {len(failures)} failed"
    )

    if failures:
        print()
        print("Failures:")
        for ticker, accession, exc in failures:
            print(f"  {ticker} {accession}: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
