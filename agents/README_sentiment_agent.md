# Sentiment Agent — Task 11

This module implements Task 11: the Sentiment Agent.

The Sentiment Agent is a Stage 1 cooperative agent. Its job is to summarize market sentiment for a company using cutoff-safe news and social media inputs. It reads company news from Task 5 and Reddit posts from Task 6, then produces a structured `SentimentReport` that follows the shared Task 20 contract.

## Purpose

The goal of this agent is not to make the final stock prediction. Instead, it provides one of the input reports that the Leader Agent will later use when making the buy / not-buy and target-price prediction.

The sentiment task is completeness-oriented rather than decision-oriented. Its goal is to summarize the sentiment evidence available before the T₀ cutoff, not to make a buy / not-buy recommendation.

If the market evidence is mixed, the agent should report that disagreement faithfully instead of forcing a fully bullish or bearish conclusion.

## Inputs

The agent takes:

* `ticker`: stock ticker, such as `"AAPL"`
* `news`: list of cutoff-safe FinnHub news articles from Task 5
* `social`: list of cutoff-safe Reddit posts from Task 6
* `client`: shared `LLMClient` used by the agent layer

The news and social inputs are assumed to have already been filtered by the T₀ cutoff timestamp. This means the agent should never see articles or posts from after the T₀ market close.

## Output

The agent returns a `SentimentReport` matching the Task 20 shared contract:

```python
{
    "ticker": str,
    "summary": str,
    "signal": "bullish" | "bearish" | "neutral" | "mixed",
    "confidence": float,
    "news_count": int,
    "social_count": int,
    "disagreement": bool,
    "citations": list[str],
}
```

The `signal` field summarizes the overall sentiment:

* `"bullish"`: mostly positive market sentiment
* `"bearish"`: mostly negative market sentiment
* `"neutral"`: little clear directional sentiment
* `"mixed"`: conflicting signals across news/social sources

The `disagreement` field should be `True` when the evidence contains meaningful conflict, such as positive news coverage but cautious or negative Reddit discussion.

## Files

```text
agents/
├── sentiment_agent.py
├── run_sentiment_agent.py
└── test_sentiment_agent.py
```

## `sentiment_agent.py`

This file contains the core Task 11 logic.

Main responsibilities:

1. Build compact evidence packets from news and Reddit inputs.
2. Prompt the LLM to summarize sentiment.
3. Require JSON output matching `SentimentReport`.
4. Parse the model response leniently.
5. Normalize the report so it conforms to the Task 20 contract.

Important implementation details:

* The agent does not call FinnHub or Reddit directly.
* The agent does not make buy / not-buy recommendations.
* The agent does not predict a target price.
* The parser clamps `confidence` to `[0, 1]`.
* The parser only accepts valid sentiment signals:

  * `"bullish"`
  * `"bearish"`
  * `"neutral"`
  * `"mixed"`
* The parser fills `news_count` and `social_count` from the actual inputs instead of trusting the model-generated counts.

## `run_sentiment_agent.py`

This file runs the Sentiment Agent standalone for one ticker.

It:

1. Loads settings from `.env`.
2. Fetches cutoff-safe FinnHub news using Task 5.
3. Fetches cutoff-safe Reddit posts using Task 6.
4. Runs the Sentiment Agent using `LLMClient`.
5. Saves:

   * `transcript.json`
   * `transcript.md`
   * `sentiment_report.json`

Example usage from the repo root:

```bash
uv run python agents/run_sentiment_agent.py AAPL 2025-11-03T16:00:00-05:00
```

The timestamp argument is the T₀ cutoff timestamp. In the full pipeline, this value should come from the Task 2 `T0Window` stored in `PipelineState`.

## `test_sentiment_agent.py`

The tests run offline and do not call any external APIs or live LLMs.

They use a scripted fake LLM client to verify:

* news evidence formatting
* Reddit evidence formatting
* prompt construction
* JSON extraction from plain JSON, fenced JSON, and prose-wrapped JSON
* valid `SentimentReport` parsing
* invalid signal fallback to `"neutral"`
* confidence clamping
* string-to-boolean handling for `disagreement`
* end-to-end agent execution
* preservation of conflicting news/social sentiment

## Leakage Protection

The Sentiment Agent assumes the T₀ cutoff has already been enforced by the retrieval layer.

Leakage protection happens upstream:

* Task 5 filters FinnHub news at or before the T₀ cutoff.
* Task 6 filters Reddit posts at or before the T₀ cutoff.

As a result, the Sentiment Agent only receives cutoff-safe inputs and never has direct access to information published after the T₀ market close.

This separation keeps the agent logic simple, makes the data boundary explicit, and ensures that all sentiment analysis is based solely on information that would have been available at prediction time.

## Relationship to the Pipeline

The Sentiment Agent produces the `sentiment_report` field in `PipelineState`.

Its output is later consumed by the Leader Agent together with:

* `fundamental_report`
* `risk_assessment`

The Leader then combines the fundamental, sentiment, and risk reports using its own judgment process to produce the initial buy / not-buy and target-price prediction.

