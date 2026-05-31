# Qualitative Risk Agent — Task 12

This module implements Task 12: the Qualitative Risk Agent.

The Qualitative Risk Agent is a Stage 1 risk-analysis agent. Its job is to analyze risk from a company’s 10-K filing and produce a structured `RiskScore` that follows the shared Task 20 contract.

## Purpose

The goal of this agent is not to make the final stock prediction. Instead, it provides one side of the project’s competitive risk-analysis process.

The qualitative risk task is text-oriented rather than financial-model-oriented. Its purpose is to identify and assess material risks described in the company’s 10-K filing, especially the Risk Factors section, and summarize those risks in a structured form.

The agent does not make buy / not-buy recommendations and does not predict target prices.

## Inputs

The agent takes:

* `ticker`: stock ticker, such as `"AAPL"`
* `retrieval_tool`: Task 7 10-K RAG retrieval callable
* `client`: shared `LLMClient` used by the agent layer

The retrieval tool is expected to be created from the Task 7 RAG system and provides semantic access to the company’s 10-K filing.

## Output

The agent returns a `RiskScore` matching the Task 20 shared contract:

```python
{
    "method": "qualitative",
    "score": float,
    "summary": str,
    "factors": list[str],
    "justification": str,
}
```

The `score` field uses a 0–10 scale:

* `0` = lowest risk
* `10` = highest risk

The parser clamps scores to the range `[0, 10]` and always forces the method field to `"qualitative"`.

## Files

```text
agents/
├── qualitative_risk_agent.py
├── run_qualitative_risk_agent.py
└── test_qualitative_risk_agent.py
```

## How It Works

The agent follows the same general architecture as the Fundamental Agent.

The model receives access to a `search_10k` retrieval tool built from Task 7 and uses that tool to gather evidence from the filing before producing a final risk assessment.

The agent is encouraged to begin with Item 1A Risk Factors but may retrieve information from other sections when relevant.

The model is prompted to consider:

* Risk Factors (Item 1A)
* litigation risk
* competition risk
* regulatory risk
* cybersecurity risk
* supply chain risk
* operational risk
* other material risks disclosed in the filing

The final output is a structured `RiskScore`.

## Leakage Protection

The Qualitative Risk Agent operates exclusively on the company’s 10-K filing through the Task 7 retrieval system.

It does not retrieve news, social-media content, market prices, or any information published after the filing.

As a result, the agent only receives filing-based information and never has direct access to information from the prediction window.

This separation keeps the risk analysis grounded in the company’s own disclosures and ensures that the agent evaluates only information available in the filing itself.

## `qualitative_risk_agent.py`

This file contains the core Task 12 logic.

Main responsibilities:

1. Build the `search_10k` tool wrapper.
2. Construct the qualitative-risk prompt.
3. Run the model through the retrieval loop.
4. Extract JSON from the model response.
5. Normalize the output into a valid `RiskScore`.

Important implementation details:

* The agent uses `run_tool_loop`, following the same pattern as the Fundamental Agent.
* The retrieval tool supports section filtering, allowing the model to request specific sections such as Item 1A Risk Factors.
* The parser clamps scores to `[0, 10]`.
* The parser always forces `method="qualitative"`.
* The parser defaults malformed scores to `5.0`.
* The agent does not make buy / not-buy recommendations.
* The agent does not predict target prices.

## `run_qualitative_risk_agent.py`

This file runs the Qualitative Risk Agent standalone for one ticker.

It:

1. Loads settings from `.env`.
2. Finds the cached 10-K filing.
3. Builds or loads the Task 7 RAG index.
4. Creates the retrieval tool.
5. Runs the Qualitative Risk Agent using `LLMClient`.
6. Saves:

   * `transcript.json`
   * `transcript.md`
   * `qualitative_risk_score.json`

Example usage from the repository root:

```bash
uv run --extra rag python agents/run_qualitative_risk_agent.py AAPL
```

Before running the agent, make sure the EDGAR filing and RAG index have already been created.

## `test_qualitative_risk_agent.py`

The tests run entirely offline and do not call any external APIs or live LLMs.

They use a scripted fake LLM client and a fake retrieval tool to verify:

* prompt construction
* retrieval-tool wrapping
* JSON extraction from plain JSON, fenced JSON, and prose-wrapped JSON
* valid `RiskScore` parsing
* forced `"qualitative"` method values
* score clamping to `[0, 10]`
* malformed-score fallback behavior
* end-to-end tool-loop execution

## Relationship to the Pipeline

The Qualitative Risk Agent produces a `RiskScore` with:

```python
"method": "qualitative"
```

This score is later paired with the quantitative risk score produced by Task 13.

Task 14 combines the two risk perspectives into a `RiskAssessment`, allowing the system to compare text-based and financial-data-based views of company risk before the Leader Agent makes its prediction.

The Leader Agent later considers:

* `fundamental_report`
* `sentiment_report`
* `risk_assessment`

when producing the initial buy / not-buy and target-price prediction.

