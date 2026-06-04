"""
Quantitative Risk Agent — Task 13.

This agent analyzes risk from structured financial data (Task 4) and,
optionally, supplementary 10-K context via the RAG retrieval tool
(Task 7).  It builds a weighted risk model over quantitative metrics
(leverage, profitability, cash flow, valuation) and produces a
quantitative RiskScore matching the Task 20 contract.

The quantitative agent is the data-oriented counterpart to the
Qualitative Risk Agent (Task 12).  Their two RiskScores are later
combined into a single RiskAssessment (Task 14) for the Leader.
"""

from __future__ import annotations

import json
import re

from llm_client import LLMClient, Tool, run_tool_loop
from schemas import RiskScore


DEFAULT_SCORE = 5.0
METHOD = "quantitative"


SYSTEM_PROMPT = """\
You are a quantitative risk analyst.

Your job is to assess company risk by building a weighted risk model \
from structured financial data. You have access to two tools:

  - get_financials: returns the company's structured financial metrics \
(profitability, cash flow, debt/leverage, and valuation ratios). Call \
this first.
  - search_10k: semantic search over the company's 10-K filing. Use \
this to find MD&A discussion or context that helps you calibrate the \
quantitative risk model (e.g. management commentary on debt covenants, \
capital allocation, or margin pressure).

Build a weighted risk model that evaluates these dimensions:

  1. **Leverage / Solvency** — debt-to-equity ratio, current ratio, \
interest coverage. High leverage or thin coverage raises risk.
  2. **Profitability** — margins (gross, operating, net), ROE, ROA. \
Weak or declining profitability raises risk.
  3. **Cash Flow** — operating cash flow, free cash flow, capex burden. \
Negative or deteriorating cash flow raises risk.
  4. **Valuation** — P/E, P/B, P/S, EV/EBITDA. Stretched valuations \
raise risk because they leave less room for error.

Assign weights to each dimension based on the company's situation and \
explain your weighting. Missing or unavailable metrics should be noted \
and handled gracefully (they raise uncertainty, which is itself a risk \
factor).

In your justification, reference the specific financial metrics and \
10-K sections that most influenced your risk assessment.

Do not make a buy/not-buy recommendation.
Do not predict a target price.
Do not use information outside the financials and 10-K.

When complete, output ONLY a single JSON object with exactly these fields:

{
  "method": "quantitative",
  "score": <number from 0 to 10, where 10 = highest risk>,
  "summary": "<2-4 sentence quantitative risk summary>",
  "factors": ["<risk factor>", "..."],
  "justification": "<why this score is appropriate, referencing metrics>"
}
"""


def build_financials_tool(financials: dict) -> Tool:
    """
    Wrap the Task 4 Financials dict as an LLM tool.

    The model calls get_financials() to inspect the company's structured
    financial data (profitability, cash flow, debt, valuation).
    """
    financials_json = json.dumps(financials, indent=2, default=str)

    def impl(**kwargs):
        return financials_json

    return Tool(
        name="get_financials",
        description=(
            "Retrieve the company's structured financial metrics: "
            "profitability (revenue, margins, ROE, ROA), cash flow "
            "(operating, capex, free), debt/leverage (debt-to-equity, "
            "current ratio, interest coverage), and valuation ratios "
            "(P/E, P/B, P/S, EV/EBITDA). Takes no arguments."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        impl=impl,
    )


def build_search_tool(retrieval_tool) -> Tool:
    """
    Wrap the Task 7 10-K retrieval callable as an LLM tool.

    This gives the quantitative agent supplementary access to the 10-K
    for context (e.g. MD&A discussion of leverage, margin drivers).
    """

    def impl(query, k=5, section=None):
        return retrieval_tool(query, k=k, section=section)

    return Tool(
        name="search_10k",
        description=(
            "Semantic search over the company's 10-K. Use section='7' "
            "for MD&A or '1A' for Risk Factors. Returns relevant chunks "
            "with chunk ids, section labels, and similarity scores."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Financial-risk query to search in the 10-K.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return.",
                },
                "section": {
                    "type": "string",
                    "description": "Optional 10-K section code, e.g. '7' or '1A'.",
                },
            },
            "required": ["query"],
        },
        impl=impl,
    )


def build_user_prompt(ticker: str, financials: dict) -> str:
    """
    Compose the initial user message: the task, a summary of available
    financial categories, and the request for a quantitative risk
    assessment.
    """
    categories = []
    for key in ("profitability", "cash_flow", "debt", "valuation"):
        section = financials.get(key, {})
        if isinstance(section, dict):
            non_null = [k for k, v in section.items() if v is not None]
            if non_null:
                categories.append(f"  - {key}: {', '.join(non_null)}")

    category_summary = "\n".join(categories) if categories else "  (none available)"

    return (
        f"Assess the quantitative risk profile of {ticker.upper()} by "
        f"building a weighted risk model from the structured financial data. "
        f"Start by calling get_financials to retrieve the metrics.\n\n"
        f"Available financial categories:\n{category_summary}\n\n"
        f"Then use search_10k to retrieve MD&A or risk-factor context as "
        f"needed to calibrate your model. Return the required RiskScore "
        f"JSON only."
    )


def extract_json_object(text: str) -> dict:
    """
    Pull the first JSON object out of model output.
    """
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("No balanced JSON object found in model output.")


def parse_risk_score(text: str) -> RiskScore:
    """
    Parse and normalize the model output into a quantitative RiskScore.
    """
    data = extract_json_object(text)

    try:
        score = float(data.get("score", DEFAULT_SCORE))
    except (TypeError, ValueError):
        score = DEFAULT_SCORE

    score = min(max(score, 0.0), 10.0)

    factors = data.get("factors")
    if not isinstance(factors, list):
        factors = []

    return RiskScore(
        method=METHOD,
        score=score,
        summary=str(data.get("summary", "")),
        factors=[str(f) for f in factors],
        justification=str(data.get("justification", "")),
    )


def run_quantitative_risk_agent(
    ticker: str,
    financials: dict,
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 8,
) -> RiskScore:
    """
    Run the quantitative risk agent end to end.

    Inputs:
        ticker:         stock ticker.
        financials:     structured financials dict (Task 4 output).
        retrieval_tool: rag_10k retrieval callable bound to this ticker's
                        10-K index (Task 7, make_retrieval_tool).
        client:         an LLMClient (Anthropic or local).
        max_iterations: tool-loop cap.

    Returns a RiskScore with method="quantitative".
    """
    tools = [build_financials_tool(financials), build_search_tool(retrieval_tool)]

    messages = [
        {
            "role": "user",
            "text": build_user_prompt(ticker, financials),
        }
    ]

    response, _ = run_tool_loop(
        client,
        messages,
        tools,
        system=SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

    return parse_risk_score(response.text)
