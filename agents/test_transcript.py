import json

from llm_client import ToolCall, LLMResponse
from transcript import TranscriptRecorder, new_run_dir, _extract_usage


def _response(text="", tool_calls=None, stop_reason="end_turn"):
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
    )


def test_record_llm_and_tool_events_in_order():
    rec = TranscriptRecorder(metadata={"ticker": "AAPL"})

    call = ToolCall(id="t1", name="search", arguments={"query": "risk"})
    rec.record_llm_call(
        "anthropic", "claude",
        request={"system": "You are an analyst."},
        raw_response={"usage": {"input_tokens": 10, "output_tokens": 3}},
        response=_response(text="let me look", tool_calls=[call],
                           stop_reason="tool_use"),
    )
    rec.record_tool_call("search", {"query": "risk"}, "chunk text")
    rec.record_llm_call(
        "anthropic", "claude",
        request={"system": "You are an analyst."},
        raw_response={"usage": {"input_tokens": 20, "output_tokens": 5}},
        response=_response(text='{"signal": "bullish"}'),
    )

    types = [e["type"] for e in rec.events]
    assert types == ["llm_call", "tool_call", "llm_call"]

    first = rec.events[0]
    assert first["tool_calls"][0]["name"] == "search"
    assert first["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert rec.events[1]["result"] == "chunk text"
    assert rec.events[1]["error"] is None


def test_capture_raw_toggle_omits_payloads():
    rec = TranscriptRecorder(capture_raw=False)

    rec.record_llm_call(
        "local", "qwen",
        request={"messages": [{"role": "system", "content": "sys"}]},
        raw_response={"usage": {"total_tokens": 7}},
        response=_response(text="hi"),
    )

    event = rec.events[0]
    assert "raw_request" not in event
    assert "raw_response" not in event
    # Normalized fields + usage are still kept.
    assert event["assistant_text"] == "hi"
    assert event["usage"] == {"total_tokens": 7}


def test_to_dict_shape():
    rec = TranscriptRecorder(metadata={"a": 1})
    rec.record_tool_call("search", {"q": "x"}, "r")

    d = rec.to_dict()
    assert d["metadata"] == {"a": 1}
    assert len(d["events"]) == 1


def test_to_markdown_includes_system_prompt_and_events():
    rec = TranscriptRecorder()
    call = ToolCall(id="t1", name="search", arguments={"query": "debt"})
    rec.record_llm_call(
        "anthropic", "claude",
        request={"system": "SYSTEM-PROMPT-MARKER"},
        raw_response={},
        response=_response(text="thinking", tool_calls=[call],
                           stop_reason="tool_use"),
    )
    rec.record_tool_call("search", {"query": "debt"}, "RESULT-MARKER")

    md = rec.to_markdown()
    assert "SYSTEM-PROMPT-MARKER" in md
    assert "search" in md
    assert "RESULT-MARKER" in md


def test_to_markdown_includes_initial_user_prompt():
    rec = TranscriptRecorder()
    rec.record_llm_call(
        "local", "qwen",
        request={
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "FINANCIALS-MARKER"},
            ]
        },
        raw_response={},
        response=_response(text="hi"),
    )

    md = rec.to_markdown()
    assert "Initial user prompt" in md
    assert "FINANCIALS-MARKER" in md


def test_save_writes_both_files_and_round_trips(tmp_path):
    rec = TranscriptRecorder(metadata={"ticker": "AAPL"})
    rec.record_tool_call("search", {"q": "x"}, "r")

    json_path, md_path = rec.save(tmp_path)

    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["ticker"] == "AAPL"
    assert loaded["events"][0]["name"] == "search"


def test_new_run_dir_creates_labeled_dir(tmp_path):
    run_dir = new_run_dir(tmp_path, "AAPL")
    assert run_dir.exists()
    assert run_dir.name.startswith("AAPL_")


def test_extract_usage_none_safe():
    assert _extract_usage({"usage": {"x": 1}}) == {"x": 1}
    assert _extract_usage({}) is None
    assert _extract_usage(None) is None
