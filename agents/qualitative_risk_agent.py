"""
Qualitative Risk Agent — Task 12.

This agent analyzes risk from the 10-K text using the RAG retrieval tool
from Task 7 and produces a qualitative RiskScore matching the Task 20
contract.
"""

from __future__ import annotations

import json
import re

from llm_client import LLMClient, Tool, run_tool_loop
from schemas import RiskScore


DEFAULT_SCORE = 5.0
METHOD = "qualitative"


SYSTEM_PROMPT = """\
You are a qualitative risk analyst.

Your job is to assess company risk using only the provided 10-K retrieval \
tool. Focus especially on Item 1A Risk Factors, but also investigate \
litigation, competition, regulation, cybersecurity, supply chain, \
operational risks, and any other material risks described in the filing.

Use the search_10k tool multiple times if necessary. Begin with Item 1A \
Risk Factors, then search for other risk-related topics if they are relevant. \
Ground your assessment in the retrieved 10-K chunks.

In your justification, reference the retrieved chunk ids or 10-K sections \
that most influenced your risk assessment.

Do not make a buy/not-buy recommendation.
Do not predict a target price.
Do not use information outside the 10-K.

When complete, output ONLY a single JSON object with exactly these fields:

{
  "method": "qualitative",
  "score": <number from 0 to 10, where 10 = highest risk>,
  "summary": "<2-4 sentence qualitative risk summary>",
  "factors": ["<risk factor>", "..."],
  "justification": "<why this score is appropriate>"
}
"""


def build_search_tool(retrieval_tool) -> Tool:
    """
    Wrap the Task 7 10-K retrieval callable as an LLM tool.
    """

    def impl(query, k=5, section=None):
        return retrieval_tool(query, k=k, section=section)

    return Tool(
        name="search_10k",
        description=(
            "Semantic search over the company's 10-K. Use section='1A' "
            "for Risk Factors. Returns relevant chunks with chunk ids, "
            "section labels, and similarity scores."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Risk-related query to search in the 10-K.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return.",
                },
                "section": {
                    "type": "string",
                    "description": "Optional 10-K section code, e.g. '1A'.",
                },
            },
            "required": ["query"],
        },
        impl=impl,
    )


def build_user_prompt(ticker: str) -> str:
    return (
        f"Assess the qualitative risk profile of {ticker.upper()} using "
        f"the 10-K search tool. Start with Item 1A Risk Factors, then "
        f"search for litigation, competition, regulation, cybersecurity, "
        f"supply chain, and operational risks as needed. Use the retrieved "
        f"chunk ids or 10-K sections in your justification. Return the "
        f"required RiskScore JSON only."
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
    Parse and normalize the model output into a qualitative RiskScore.
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


def run_qualitative_risk_agent(
    ticker: str,
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 8,
) -> RiskScore:
    """
    Run the qualitative risk agent end to end.
    """
    tools = [build_search_tool(retrieval_tool)]

    messages = [
        {
            "role": "user",
            "text": build_user_prompt(ticker),
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
