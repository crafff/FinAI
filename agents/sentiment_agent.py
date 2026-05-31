"""
Sentiment agent (Task 11, Stage 1 - cooperative).

A single agent that summarizes market sentiment from company news
(FinnHub) and social media posts (Reddit/PRAW) and emits a
SentimentReport matching contracts/schemas.py.

Leakage: this agent does not fetch live data directly. It receives
already-filtered news and social posts from Tasks 5 and 6, which enforce
the T0 cutoff timestamp before the agent sees any inputs.
"""

from __future__ import annotations

import json
import re

from llm_client import LLMClient
from schemas import SentimentReport


VALID_SIGNALS = ("bullish", "bearish", "neutral", "mixed")
DEFAULT_SIGNAL = "neutral"
DEFAULT_CONFIDENCE = 0.5


SYSTEM_PROMPT = """\
You are a market sentiment analyst. Your job is to assess market sentiment \
for one company using only the provided pre-cutoff news articles and Reddit \
posts.

The inputs have already been filtered to include only information available \
at or before the T0 market-close cutoff. Do not assume knowledge of events \
after the provided evidence.

Your task is cooperative and completeness-oriented: summarize the sentiment \
faithfully. If news and social media conflict, or if the evidence contains \
both positive and negative signals, report that disagreement instead of \
forcing a single clean narrative.

When your analysis is complete, output ONLY a single JSON object with exactly \
these fields:

  {
    "summary": "<2-4 sentence sentiment assessment>",
    "signal": "bullish" | "bearish" | "neutral" | "mixed",
    "confidence": <number between 0 and 1>,
    "news_count": <integer count of provided news articles>,
    "social_count": <integer count of provided social posts>,
    "disagreement": <true or false>,
    "citations": ["<news/source/post ids you relied on>", ...]
  }

Use the evidence ids such as "news_1" and "social_1" in citations. Do not \
make a buy/not-buy recommendation and do not predict a target price.
"""


def _shorten(text, max_chars=500):
    if text is None:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def build_news_evidence(news, max_items=12):
    """
    Convert NewsItem records into compact evidence strings.
    """
    evidence = []

    for idx, item in enumerate(news[:max_items], start=1):
        source = item.get("source") or "unknown source"
        headline = _shorten(item.get("headline"), 200)
        summary = _shorten(item.get("summary"), 500)
        published = item.get("published_at_et")
        url = item.get("url")

        evidence.append({
            "id": f"news_{idx}",
            "source": source,
            "published_at_et": str(published),
            "headline": headline,
            "summary": summary,
            "url": url,
        })

    return evidence


def build_social_evidence(social, max_items=12):
    """
    Convert RedditPost records into compact evidence strings.
    """
    evidence = []

    for idx, post in enumerate(social[:max_items], start=1):
        title = _shorten(post.get("title"), 200)
        body = _shorten(post.get("body"), 500)
        subreddit = post.get("subreddit") or "unknown subreddit"
        score = post.get("score")
        comments = post.get("num_comments")
        published = post.get("published_at_et")
        permalink = post.get("permalink")
        post_id = post.get("id")

        evidence.append({
            "id": f"social_{idx}",
            "reddit_id": post_id,
            "subreddit": subreddit,
            "score": score,
            "num_comments": comments,
            "published_at_et": str(published),
            "title": title,
            "body": body,
            "permalink": permalink,
        })

    return evidence


def build_user_prompt(ticker: str, news: list[dict], social: list[dict]) -> str:
    """
    Compose the initial user message containing cutoff-safe evidence.
    """
    news_evidence = build_news_evidence(news)
    social_evidence = build_social_evidence(social)

    payload = {
        "ticker": ticker.upper(),
        "news_count": len(news),
        "social_count": len(social),
        "news_evidence": news_evidence,
        "social_evidence": social_evidence,
    }

    return (
        f"Assess market sentiment for {ticker.upper()} using the evidence below.\n\n"
        f"Evidence JSON:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Return the required JSON SentimentReport fields only."
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


def parse_sentiment_report(
    text: str,
    ticker: str,
    news_count: int,
    social_count: int,
) -> SentimentReport:
    """
    Parse and normalize the model's final JSON into a SentimentReport.

    The parser fills news_count and social_count from the actual inputs
    rather than trusting the model, so the counts always match the
    cutoff-safe evidence passed to the agent.
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

    disagreement = data.get("disagreement", False)
    if not isinstance(disagreement, bool):
        disagreement = str(disagreement).lower() == "true"

    citations = data.get("citations")
    if not isinstance(citations, list):
        citations = []

    return SentimentReport(
        ticker=ticker.upper(),
        summary=str(data.get("summary", "")),
        signal=signal,
        confidence=confidence,
        news_count=int(news_count),
        social_count=int(social_count),
        disagreement=disagreement,
        citations=[str(c) for c in citations],
    )


def run_sentiment_agent(
    ticker: str,
    news: list[dict],
    social: list[dict],
    client: LLMClient,
) -> SentimentReport:
    """
    Run the sentiment agent end to end.

    Inputs:
        ticker: stock ticker.
        news: cutoff-safe FinnHub NewsItem records from Task 5.
        social: cutoff-safe RedditPost records from Task 6.
        client: an LLMClient.

    Returns:
        SentimentReport.
    """
    messages = [
        {
            "role": "user",
            "text": build_user_prompt(ticker, news, social),
        }
    ]

    response = client.complete(
        messages,
        tools=None,
        system=SYSTEM_PROMPT,
    )

    return parse_sentiment_report(
        response.text,
        ticker=ticker,
        news_count=len(news),
        social_count=len(social),
    )
