from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from llm_client import LLMResponse
from sentiment_agent import (
    build_news_evidence,
    build_social_evidence,
    build_user_prompt,
    extract_json_object,
    parse_sentiment_report,
    run_sentiment_agent,
)


NY = ZoneInfo("America/New_York")


class ScriptedClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
        })

        return LLMResponse(
            text=self.response_text,
            tool_calls=[],
            stop_reason="end_turn",
        )


def _news_item(headline="Good news", summary="Strong demand"):
    return {
        "headline": headline,
        "summary": summary,
        "source": "Test News",
        "url": "https://example.com/news",
        "published_at_et": datetime(2026, 2, 3, 15, 0, tzinfo=NY),
        "published_unix": 1770148800,
    }


def _reddit_post(title="Bullish post", body="Investors seem optimistic"):
    return {
        "id": "abc123",
        "title": title,
        "body": body,
        "subreddit": "stocks",
        "score": 42,
        "num_comments": 7,
        "url": "https://reddit.com/example",
        "permalink": "/r/stocks/comments/abc123",
        "published_at_et": datetime(2026, 2, 3, 15, 30, tzinfo=NY),
        "published_unix": 1770150600,
    }


def test_build_news_evidence_uses_expected_fields():
    evidence = build_news_evidence([_news_item()])

    assert len(evidence) == 1
    assert evidence[0]["id"] == "news_1"
    assert evidence[0]["headline"] == "Good news"
    assert evidence[0]["summary"] == "Strong demand"
    assert evidence[0]["source"] == "Test News"


def test_build_social_evidence_uses_expected_fields():
    evidence = build_social_evidence([_reddit_post()])

    assert len(evidence) == 1
    assert evidence[0]["id"] == "social_1"
    assert evidence[0]["reddit_id"] == "abc123"
    assert evidence[0]["title"] == "Bullish post"
    assert evidence[0]["subreddit"] == "stocks"


def test_build_user_prompt_contains_counts_and_ticker():
    prompt = build_user_prompt(
        "aapl",
        news=[_news_item()],
        social=[_reddit_post()],
    )

    assert "AAPL" in prompt
    assert '"news_count": 1' in prompt
    assert '"social_count": 1' in prompt
    assert "Good news" in prompt
    assert "Bullish post" in prompt


def test_extract_json_object_plain_json():
    result = extract_json_object('{"signal": "bullish"}')

    assert result["signal"] == "bullish"


def test_extract_json_object_with_fence():
    result = extract_json_object(
        '```json\n{"signal": "bearish"}\n```'
    )

    assert result["signal"] == "bearish"


def test_extract_json_object_with_surrounding_prose():
    result = extract_json_object(
        'Here is the result: {"signal": "mixed", "confidence": 0.6}'
    )

    assert result["signal"] == "mixed"
    assert result["confidence"] == 0.6


def test_parse_sentiment_report_valid_output():
    text = """
    {
      "summary": "News is positive but Reddit is cautious.",
      "signal": "mixed",
      "confidence": 0.72,
      "disagreement": true,
      "citations": ["news_1", "social_1"]
    }
    """

    result = parse_sentiment_report(
        text,
        ticker="aapl",
        news_count=3,
        social_count=2,
    )

    assert result["ticker"] == "AAPL"
    assert result["summary"] == "News is positive but Reddit is cautious."
    assert result["signal"] == "mixed"
    assert result["confidence"] == pytest.approx(0.72)
    assert result["news_count"] == 3
    assert result["social_count"] == 2
    assert result["disagreement"] is True
    assert result["citations"] == ["news_1", "social_1"]


def test_parse_sentiment_report_defaults_bad_signal_and_clamps_confidence():
    text = """
    {
      "summary": "Unclear sentiment.",
      "signal": "very bullish",
      "confidence": 1.7,
      "disagreement": false,
      "citations": "news_1"
    }
    """

    result = parse_sentiment_report(
        text,
        ticker="MSFT",
        news_count=1,
        social_count=0,
    )

    assert result["signal"] == "neutral"
    assert result["confidence"] == 1.0
    assert result["citations"] == []


def test_parse_sentiment_report_handles_string_disagreement():
    text = """
    {
      "summary": "Conflicting signals.",
      "signal": "mixed",
      "confidence": 0.5,
      "disagreement": "true",
      "citations": []
    }
    """

    result = parse_sentiment_report(
        text,
        ticker="MSFT",
        news_count=1,
        social_count=1,
    )

    assert result["disagreement"] is True


def test_run_sentiment_agent_returns_report():
    client = ScriptedClient("""
    {
      "summary": "Sentiment is broadly positive.",
      "signal": "bullish",
      "confidence": 0.8,
      "disagreement": false,
      "citations": ["news_1"]
    }
    """)

    result = run_sentiment_agent(
        ticker="aapl",
        news=[_news_item()],
        social=[_reddit_post()],
        client=client,
    )

    assert result["ticker"] == "AAPL"
    assert result["signal"] == "bullish"
    assert result["confidence"] == pytest.approx(0.8)
    assert result["news_count"] == 1
    assert result["social_count"] == 1

    assert len(client.calls) == 1
    assert client.calls[0]["tools"] is None
    assert client.calls[0]["system"] is not None

def test_run_sentiment_agent_reports_conflicting_signals():
    client = ScriptedClient("""
    {
      "summary": "News coverage is positive, but Reddit discussion is cautious.",
      "signal": "mixed",
      "confidence": 0.65,
      "disagreement": true,
      "citations": ["news_1", "social_1"]
    }
    """)

    result = run_sentiment_agent(
        ticker="aapl",
        news=[
            _news_item(
                headline="Apple reports strong demand",
                summary="Analysts describe demand as resilient."
            )
        ],
        social=[
            _reddit_post(
                title="AAPL valuation looks stretched",
                body="Retail investors are worried the stock is expensive."
            )
        ],
        client=client,
    )

    assert result["ticker"] == "AAPL"
    assert result["signal"] == "mixed"
    assert result["disagreement"] is True
    assert result["news_count"] == 1
    assert result["social_count"] == 1
    assert result["citations"] == ["news_1", "social_1"]
