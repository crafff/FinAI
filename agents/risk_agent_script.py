"""
Three-Phase Risk Pipeline (Task 14).

Orchestrates the Qualitative Risk Agent (Task 12) and the Quantitative
Risk Agent (Task 13) through a structured three-phase protocol that
maps directly to the RiskAssessment contract (contracts/schemas.py):

    Phase 1 — Cooperate:  Both agents share tools to cooperatively
                          enumerate ALL material risk factors from the
                          10-K filing and structured financial data.
                          Output: collected_factors.

    Phase 2 — Compete:    Each agent independently scores risk, informed
                          by the shared Phase 1 factors but applying its
                          own lens (qualitative/textual vs. quantitative/
                          numerical). They produce OPPOSING scores.
                          Output: two independent RiskScores.

    Phase 3 — Submit:     The two opposing RiskScores are assembled into
                          a RiskAssessment for the Leader (Task 15).
                          Output: RiskAssessment.

The output is a RiskAssessment matching the Task 20 contract:

    {
        "collected_factors": [...],        # Phase 1 cooperative list
        "scores": [
            {"method": "qualitative",  ...},   # Phase 2 opposing pair
            {"method": "quantitative", ...},
        ],
    }

Usage from the repo root:

    uv run --extra rag python agents/risk_agent_script.py AAPL
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _sub in (
    "config",
    "contracts",
    "data/EDGAR_retrieval",
    "data/financial_retrieval",
    "data/rag_10k",
):
    sys.path.insert(0, str(ROOT / _sub))

from llm_client import LLMClient, run_tool_loop  # noqa: E402
from schemas import RiskAssessment, RiskScore  # noqa: E402

from qualitative_risk_agent import (  # noqa: E402
    build_search_tool,
    extract_json_object,
    parse_risk_score as parse_qualitative_score,
)
from quantitative_risk_agent import (  # noqa: E402
    build_financials_tool,
    parse_risk_score as parse_quantitative_score,
)


# =====================================================================
# Phase 1 — Cooperate: joint factor enumeration
# =====================================================================

PHASE1_SYSTEM_PROMPT = """\
You are a risk analyst performing the first phase of a structured risk \
assessment. Your job is to cooperatively enumerate ALL material risk \
factors for the company, drawing on both qualitative (10-K text) and \
quantitative (structured financial metrics) sources.

You have two tools:

  - get_financials: returns the company's structured financial metrics \
(profitability, cash flow, debt/leverage, valuation ratios). Call \
this first.
  - search_10k: semantic search over the company's 10-K filing. Use \
section codes like '1A' (Risk Factors), '7' (MD&A), '1' (Business).

Use BOTH tools to build a comprehensive factor list. Search broadly: \
risk factors from the 10-K, financial-statement red flags (high \
leverage, thin margins, negative cash flow, stretched valuations), \
plus litigation, competition, regulation, cybersecurity, supply chain, \
and operational risks.

For each factor, keep the description concise (one line). Do not score \
or rank the factors. Do not make buy/not-buy recommendations. Simply \
enumerate them.

When complete, output ONLY a single JSON object:

{
  "factors": [
    "<concise risk factor description>",
    "..."
  ]
}
"""


def phase1_collect_factors(
    ticker: str,
    financials: dict,
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 6,
) -> list[str]:
    """
    Phase 1: cooperative factor enumeration.

    Both agents' tools are available so the model can draw on the 10-K
    text AND the structured financial data to build a comprehensive
    list of material risk factors.
    """
    tools = [
        build_financials_tool(financials),
        build_search_tool(retrieval_tool),
    ]

    messages = [
        {
            "role": "user",
            "text": (
                f"Enumerate all material risk factors for {ticker.upper()} "
                f"using both the structured financials and the 10-K filing. "
                f"Start by calling get_financials to inspect the numbers, "
                f"then search the 10-K for Item 1A Risk Factors, litigation, "
                f"competition, regulation, cybersecurity, supply chain, and "
                f"operational risks. Return the JSON factors list."
            ),
        }
    ]

    response, _ = run_tool_loop(
        client,
        messages,
        tools,
        system=PHASE1_SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

    data = extract_json_object(response.text)
    factors = data.get("factors", [])

    if not isinstance(factors, list):
        return []

    return [str(f) for f in factors if f]


# =====================================================================
# Phase 2 — Compete: independent opposing scores
# =====================================================================

PHASE2_QUALITATIVE_SYSTEM = """\
You are a qualitative risk analyst performing the second phase of a \
structured risk assessment. In Phase 1, a cooperative enumeration \
identified these material risk factors:

{factors_block}

Your job is to INDEPENDENTLY score the company's risk from a QUALITATIVE \
perspective, focusing on the 10-K text. Use the search_10k tool to \
gather evidence for or against each factor. You may also identify \
factors the Phase 1 enumeration missed.

Ground your assessment in the retrieved 10-K chunks. Reference the \
chunk ids or 10-K sections that most influenced your scoring.

Do not make a buy/not-buy recommendation.
Do not predict a target price.
Do not use information outside the 10-K.

When complete, output ONLY a single JSON object:

{{
  "method": "qualitative",
  "score": <number from 0 to 10, where 10 = highest risk>,
  "summary": "<2-4 sentence qualitative risk summary>",
  "factors": ["<risk factor>", "..."],
  "justification": "<why this score, referencing 10-K sections>"
}}
"""


PHASE2_QUANTITATIVE_SYSTEM = """\
You are a quantitative risk analyst performing the second phase of a \
structured risk assessment. In Phase 1, a cooperative enumeration \
identified these material risk factors:

{factors_block}

Your job is to INDEPENDENTLY score the company's risk from a \
QUANTITATIVE perspective, building a weighted risk model from the \
structured financial data. Use get_financials to retrieve the metrics, \
and optionally search_10k for MD&A context.

Build a weighted risk model that evaluates these dimensions:

  1. **Leverage / Solvency** — debt-to-equity, current ratio, interest \
coverage. High leverage or thin coverage raises risk.
  2. **Profitability** — margins (gross, operating, net), ROE, ROA. \
Weak or declining profitability raises risk.
  3. **Cash Flow** — operating cash flow, free cash flow, capex burden. \
Negative or deteriorating cash flow raises risk.
  4. **Valuation** — P/E, P/B, P/S, EV/EBITDA. Stretched valuations \
raise risk because they leave less room for error.

Assign weights to each dimension and explain your weighting. Reference \
the specific financial metrics that most influenced your scoring.

Do not make a buy/not-buy recommendation.
Do not predict a target price.

When complete, output ONLY a single JSON object:

{{
  "method": "quantitative",
  "score": <number from 0 to 10, where 10 = highest risk>,
  "summary": "<2-4 sentence quantitative risk summary>",
  "factors": ["<risk factor>", "..."],
  "justification": "<why this score, referencing metrics and weights>"
}}
"""


def phase2_score_qualitative(
    ticker: str,
    collected_factors: list[str],
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 6,
) -> RiskScore:
    """
    Phase 2a: independent qualitative risk scoring.

    The qualitative agent evaluates the collected factors through the
    lens of the 10-K text, using search_10k. It produces an opposing
    score grounded in the company's own disclosures.
    """
    tools = [build_search_tool(retrieval_tool)]

    factors_block = json.dumps(collected_factors, indent=2)
    system = PHASE2_QUALITATIVE_SYSTEM.format(factors_block=factors_block)

    messages = [
        {
            "role": "user",
            "text": (
                f"Score the qualitative risk of {ticker.upper()} by "
                f"evaluating the collected risk factors against the 10-K "
                f"filing. Use search_10k to gather evidence for or against "
                f"each factor. Return the required RiskScore JSON only."
            ),
        }
    ]

    response, _ = run_tool_loop(
        client,
        messages,
        tools,
        system=system,
        max_iterations=max_iterations,
    )

    return parse_qualitative_score(response.text)


def phase2_score_quantitative(
    ticker: str,
    collected_factors: list[str],
    financials: dict,
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 6,
) -> RiskScore:
    """
    Phase 2b: independent quantitative risk scoring.

    The quantitative agent evaluates the collected factors through the
    lens of structured financial data, building a weighted risk model.
    It produces an opposing score grounded in financial metrics.
    """
    tools = [
        build_financials_tool(financials),
        build_search_tool(retrieval_tool),
    ]

    factors_block = json.dumps(collected_factors, indent=2)
    system = PHASE2_QUANTITATIVE_SYSTEM.format(factors_block=factors_block)

    messages = [
        {
            "role": "user",
            "text": (
                f"Score the quantitative risk of {ticker.upper()} by "
                f"building a weighted model from the financial data. Call "
                f"get_financials first, then optionally search_10k for "
                f"context. Return the required RiskScore JSON only."
            ),
        }
    ]

    response, _ = run_tool_loop(
        client,
        messages,
        tools,
        system=system,
        max_iterations=max_iterations,
    )

    return parse_quantitative_score(response.text)


# =====================================================================
# Phase 3 — Submit: assemble the RiskAssessment
# =====================================================================

def phase3_assemble(
    collected_factors: list[str],
    qualitative_score: RiskScore,
    quantitative_score: RiskScore,
) -> RiskAssessment:
    """
    Phase 3: combine the cooperative factors and opposing scores into
    the final RiskAssessment matching the Task 20 contract.

    The two scores are deliberately NOT reduced to a single number.
    They are carried forward as an opposing pair for the Leader to
    reconcile with free judgment.
    """
    return RiskAssessment(
        collected_factors=collected_factors,
        scores=[qualitative_score, quantitative_score],
    )


# =====================================================================
# Orchestrator
# =====================================================================

def run_three_phase_risk_pipeline(
    ticker: str,
    financials: dict,
    retrieval_tool,
    client: LLMClient,
) -> RiskAssessment:
    """
    Run the full three-phase risk protocol end to end.

    Phase 1 — Cooperate:  enumerate all material risk factors.
    Phase 2 — Compete:    qualitative + quantitative agents score
                          independently.
    Phase 3 — Submit:     assemble the RiskAssessment.

    Returns a RiskAssessment containing the shared factors and two
    opposing RiskScores.
    """
    print("  Phase 1 — Cooperative factor collection ...", flush=True)
    collected_factors = phase1_collect_factors(
        ticker, financials, retrieval_tool, client,
    )
    print(f"    → {len(collected_factors)} factors collected", flush=True)

    print("  Phase 2a — Qualitative risk scoring ...", flush=True)
    qualitative_score = phase2_score_qualitative(
        ticker, collected_factors, retrieval_tool, client,
    )
    print(
        f"    → qualitative score: {qualitative_score['score']:.1f}/10",
        flush=True,
    )

    print("  Phase 2b — Quantitative risk scoring ...", flush=True)
    quantitative_score = phase2_score_quantitative(
        ticker, collected_factors, financials, retrieval_tool, client,
    )
    print(
        f"    → quantitative score: {quantitative_score['score']:.1f}/10",
        flush=True,
    )

    print("  Phase 3 — Assembling RiskAssessment ...", flush=True)
    assessment = phase3_assemble(
        collected_factors, qualitative_score, quantitative_score,
    )

    return assessment


# =====================================================================
# CLI entry point
# =====================================================================

EDGAR_CACHE = ROOT / ".cache" / "edgar"
RAG_CACHE = ROOT / ".cache" / "rag"


def _find_cached_filing(ticker):
    """
    Return (accession_number, text) for the cached 10-K, or None.
    """
    ticker_dir = EDGAR_CACHE / ticker.upper()
    metas = sorted(ticker_dir.glob("*.meta.json")) if ticker_dir.is_dir() else []

    if not metas:
        return None

    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    text_path = ticker_dir / f"{meta['accession_number']}.txt"

    if not text_path.exists():
        return None

    return meta["accession_number"], text_path.read_text(encoding="utf-8")


def main():
    from settings import load_settings  # noqa: E402
    from rag_10k import build_or_load_index, make_retrieval_tool  # noqa: E402
    from financial_retrieval import (  # noqa: E402
        fetch_financials,
        DEFAULT_FINANCIALS_CACHE,
    )
    from transcript import TranscriptRecorder, new_run_dir  # noqa: E402

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    cfg = load_settings()

    missing = cfg.missing("llm", "fmp")
    if missing:
        print("Missing config in .env:", ", ".join(missing))
        sys.exit(1)

    # --- data setup ---

    filing = _find_cached_filing(ticker)
    if filing is None:
        print(f"No cached 10-K for {ticker} under {EDGAR_CACHE}.")
        print("Run first: uv run python data/EDGAR_retrieval/run_fetch.py")
        sys.exit(1)

    accession, text = filing

    print(f"[1/3] RAG index for {ticker} ({accession}) ...", flush=True)
    index = build_or_load_index(
        ticker, accession, text, cache_dir=RAG_CACHE,
    )
    retrieval_tool = make_retrieval_tool(index)

    print(f"[2/3] Fetching financials for {ticker} ...", flush=True)
    financials = fetch_financials(
        ticker,
        api_key=cfg.require_fmp_api_key(),
        cache_dir=DEFAULT_FINANCIALS_CACHE,
    )

    # --- three-phase pipeline ---

    print(
        f"[3/3] Three-phase risk pipeline "
        f"({cfg.llm.backend}:{cfg.llm.model}) ...",
        flush=True,
    )

    recorder = TranscriptRecorder(metadata={
        "agent": "three_phase_risk",
        "ticker": ticker,
        "accession": accession,
        "backend": cfg.llm.backend,
        "model": cfg.llm.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    assessment = run_three_phase_risk_pipeline(
        ticker=ticker,
        financials=financials,
        retrieval_tool=retrieval_tool,
        client=LLMClient(cfg.llm, recorder=recorder),
    )

    # --- save results ---

    runs_base = Path(cfg.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    run_dir = new_run_dir(runs_base, ticker)
    recorder.save(run_dir)

    (run_dir / "risk_assessment.json").write_text(
        json.dumps(assessment, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(json.dumps(assessment, indent=2, default=str))
    print(f"\nSaved run to {run_dir}")


if __name__ == "__main__":
    main()
