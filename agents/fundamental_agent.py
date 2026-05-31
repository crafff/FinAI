"""
Fundamental agent (Task 10, Stage 1 - cooperative).

A single tool-equipped agent that investigates a company's 10-K via the
RAG retrieval tool and combines it with the structured financials
(Task 4) to produce a FundamentalReport (see contracts/schemas.py).

A single agent is used deliberately: the source paper found single agents
outperform groups on this completeness-oriented subtask, and it conserves
tokens. The agent runs the full tool-use loop - it decides when and what
to retrieve from the 10-K.

Leakage: this agent fetches no live data. The 10-K is the filing itself
and the financials are already cut off at/before the fiscal year by the
data layer, so all visible information predates the T0 cutoff.
"""

from __future__ import annotations

import json
import re

from llm_client import LLMClient, Tool, run_tool_loop
from schemas import FundamentalReport

VALID_SIGNALS = ("bullish", "bearish", "neutral", "mixed")
DEFAULT_SIGNAL = "neutral"
DEFAULT_CONFIDENCE = 0.5


SYSTEM_PROMPT = """\
You are a fundamental equity analyst. Your job is to assess one company's \
investment fundamentals from its 10-K annual report and structured \
financial metrics.

Use the `search_10k` tool to investigate the report: the business and \
competitive position (Item 1), risk factors (Item 1A), and management's \
discussion of financial condition and results (Item 7, "MD&A"). Issue \
focused queries and call the tool as many times as you need. Ground your \
analysis in what you actually retrieve and in the financial metrics \
provided; do not invent figures.

All information is as of the filing; do not assume any knowledge of events \
after it.

When your analysis is complete, output ONLY a single JSON object (no prose \
before or after it) with exactly these fields:

  {
    "summary": "<2-4 sentence fundamental assessment>",
    "signal": "bullish" | "bearish" | "neutral" | "mixed",
    "confidence": <number between 0 and 1>,
    "key_metrics": {"<name>": <number or null>, ...},
    "citations": ["<10-K section or metric you relied on>", ...]
  }
"""


def build_search_tool(retrieval_tool) -> Tool:
    """
    Wrap a rag_10k retrieval callable (from make_retrieval_tool) as a
    Tool the model can call. `retrieval_tool` has the signature
    retrieve_10k(query, k=5, section=None) -> str.
    """

    def impl(query, k=5, section=None):
        return retrieval_tool(query, k=k, section=section)

    return Tool(
        name="search_10k",
        description=(
            "Semantic search over the company's 10-K. Returns the most "
            "relevant text chunks with their section and similarity. "
            "Optionally restrict to a 10-K item code (e.g. '1' Business, "
            "'1A' Risk Factors, '7' MD&A)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for in the 10-K.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5).",
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Optional 10-K item code to restrict to, e.g. "
                        "'1A' or '7'."
                    ),
                },
            },
            "required": ["query"],
        },
        impl=impl,
    )


def build_user_prompt(ticker: str, financials: dict) -> str:
    """
    Compose the initial user message: the task plus the structured
    financials injected as context (default=str handles dates).
    """
    financials_json = json.dumps(financials, indent=2, default=str)

    return (
        f"Assess the investment fundamentals of {ticker.upper()}.\n\n"
        f"Structured financials (already cut off at the fiscal year, "
        f"point-in-time):\n{financials_json}\n\n"
        f"Investigate the 10-K with search_10k, then produce the JSON report."
    )


def extract_json_object(text: str) -> dict:
    """
    Pull the first JSON object out of model output, tolerating ```json
    fences or surrounding prose.
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


def parse_fundamental_report(text: str, ticker: str) -> FundamentalReport:
    """
    Parse and normalize the model's final JSON into a FundamentalReport.

    Lenient on the model's formatting: coerces an out-of-range signal to
    "neutral", clamps confidence to [0, 1], and defaults missing optional
    fields, so a slightly-off response still yields a valid report.
    """
    data = extract_json_object(text)

    signal = data.get("signal", DEFAULT_SIGNAL)
    if signal not in VALID_SIGNALS:
        signal = DEFAULT_SIGNAL

    try:
        confidence = float(data.get("confidence", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    confidence = min(max(confidence, 0.0), 1.0)

    key_metrics = data.get("key_metrics")
    if not isinstance(key_metrics, dict):
        key_metrics = {}

    citations = data.get("citations")
    if not isinstance(citations, list):
        citations = []

    return FundamentalReport(
        ticker=ticker.upper(),
        summary=str(data.get("summary", "")),
        signal=signal,
        confidence=confidence,
        key_metrics=key_metrics,
        citations=[str(c) for c in citations],
    )


def run_fundamental_agent(
    ticker: str,
    financials: dict,
    retrieval_tool,
    client: LLMClient,
    max_iterations: int = 8,
) -> FundamentalReport:
    """
    Run the fundamental agent end to end.

    Inputs:
        ticker:         stock ticker.
        financials:     structured financials dict (Task 4 output).
        retrieval_tool: rag_10k retrieval callable bound to this ticker's
                        10-K index (Task 7, make_retrieval_tool).
        client:         an LLMClient (Anthropic or local vLLM).
        max_iterations: tool-loop cap.

    Returns a FundamentalReport.
    """
    tools = [build_search_tool(retrieval_tool)]
    messages = [{"role": "user", "text": build_user_prompt(ticker, financials)}]

    response, _ = run_tool_loop(
        client, messages, tools,
        system=SYSTEM_PROMPT,
        max_iterations=max_iterations,
    )

    return parse_fundamental_report(response.text, ticker)
