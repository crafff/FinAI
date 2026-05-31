import json

from llm_client import (
    LLMResponse,
    Tool,
    ToolCall,
    ToolResult,
    build_anthropic_request,
    build_openai_request,
    parse_anthropic_response,
    parse_openai_response,
    run_tool_loop,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)


# --------------------------------------------------------------------------
# Tool-definition conversion
# --------------------------------------------------------------------------

TOOL_DEF = {
    "name": "search_10k",
    "description": "search",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
}


def test_to_anthropic_tools_uses_input_schema():
    result = to_anthropic_tools([TOOL_DEF])[0]

    assert result["name"] == "search_10k"
    assert result["input_schema"] == TOOL_DEF["parameters"]


def test_to_openai_tools_wraps_function():
    result = to_openai_tools([TOOL_DEF])[0]

    assert result["type"] == "function"
    assert result["function"]["name"] == "search_10k"
    assert result["function"]["parameters"] == TOOL_DEF["parameters"]


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------

def _conversation():
    return [
        {"role": "user", "text": "hi"},
        {
            "role": "assistant",
            "text": "searching",
            "tool_calls": [ToolCall(id="t1", name="search_10k", arguments={"query": "risk"})],
        },
        {"role": "tool", "tool_results": [ToolResult(id="t1", content="chunk text")]},
    ]


def test_to_anthropic_messages_shapes():
    out = to_anthropic_messages(_conversation())

    assert out[0] == {"role": "user", "content": "hi"}

    assistant = out[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "searching"}
    assert assistant["content"][1] == {
        "type": "tool_use", "id": "t1", "name": "search_10k",
        "input": {"query": "risk"},
    }

    # Tool results ride in a user-role message for Anthropic.
    tool_msg = out[2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0] == {
        "type": "tool_result", "tool_use_id": "t1", "content": "chunk text",
    }


def test_to_openai_messages_shapes():
    out = to_openai_messages(_conversation(), system="be helpful")

    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}

    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "t1"
    assert assistant["tool_calls"][0]["function"]["name"] == "search_10k"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "query": "risk"
    }

    # OpenAI uses a dedicated tool-role message.
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "chunk text"}


# --------------------------------------------------------------------------
# Request building
# --------------------------------------------------------------------------

def test_build_anthropic_request_includes_system_and_tools():
    request = build_anthropic_request(
        "claude-x", [{"role": "user", "text": "hi"}], [TOOL_DEF],
        "sys", 1000, 0.1,
    )

    assert request["model"] == "claude-x"
    assert request["max_tokens"] == 1000
    assert request["system"] == "sys"
    assert request["tools"][0]["name"] == "search_10k"


def test_build_openai_request_omits_tools_when_none():
    request = build_openai_request(
        "local-x", [{"role": "user", "text": "hi"}], None, None, 500, 0.0,
    )

    assert request["model"] == "local-x"
    assert "tools" not in request
    # System absent -> no system message.
    assert request["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

def test_parse_anthropic_response_text_and_tool_use():
    raw = {
        "content": [
            {"type": "text", "text": "let me look"},
            {"type": "tool_use", "id": "t9", "name": "search_10k",
             "input": {"query": "debt"}},
        ],
        "stop_reason": "tool_use",
    }

    response = parse_anthropic_response(raw)

    assert response.text == "let me look"
    assert response.tool_calls[0] == ToolCall("t9", "search_10k", {"query": "debt"})
    assert response.stop_reason == "tool_use"


def test_parse_openai_response_parses_arguments_json():
    raw = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "search_10k",
                                 "arguments": '{"query": "debt"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }

    response = parse_openai_response(raw)

    assert response.text == ""
    assert response.tool_calls[0].name == "search_10k"
    assert response.tool_calls[0].arguments == {"query": "debt"}


def test_parse_openai_response_plain_text():
    raw = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    response = parse_openai_response(raw)

    assert response.text == "hello"
    assert response.tool_calls == []


# --------------------------------------------------------------------------
# Tool-use loop (scripted client - no SDK)
# --------------------------------------------------------------------------

class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, system=None):
        self.calls.append({"tools": tools, "n_messages": len(messages)})
        return self.responses.pop(0)


def _echo_tool(calls):
    def impl(query, k=5, section=None):
        calls.append((query, k, section))
        return f"CHUNK[{query}]"

    return Tool(
        name="search_10k", description="d",
        parameters={"type": "object", "properties": {}},
        impl=impl,
    )


def test_run_tool_loop_executes_tool_then_returns_final():
    tool_calls_made = []
    tool = _echo_tool(tool_calls_made)

    client = ScriptedClient([
        LLMResponse(text="searching",
                    tool_calls=[ToolCall("t1", "search_10k", {"query": "risk"})]),
        LLMResponse(text='{"summary": "ok"}', tool_calls=[]),
    ])

    final, messages = run_tool_loop(client, [{"role": "user", "text": "go"}], [tool])

    assert final.text == '{"summary": "ok"}'
    # The tool was actually executed with the model's arguments.
    assert tool_calls_made == [("risk", 5, None)]
    # The tool result was fed back into the conversation.
    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_results"][0].content == "CHUNK[risk]"


def test_run_tool_loop_handles_unknown_tool():
    tool = _echo_tool([])

    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("t1", "nonexistent", {})]),
        LLMResponse(text="done", tool_calls=[]),
    ])

    final, messages = run_tool_loop(client, [{"role": "user", "text": "go"}], [tool])

    assert final.text == "done"
    assert "unknown tool" in messages[2]["tool_results"][0].content


def test_run_tool_loop_handles_tool_exception():
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    tool = Tool(name="search_10k", description="d",
                parameters={"type": "object", "properties": {}}, impl=boom)

    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("t1", "search_10k", {})]),
        LLMResponse(text="recovered", tool_calls=[]),
    ])

    final, messages = run_tool_loop(client, [{"role": "user", "text": "go"}], [tool])

    assert final.text == "recovered"
    assert "kaboom" in messages[2]["tool_results"][0].content


def test_run_tool_loop_forces_answer_at_cap():
    tool = _echo_tool([])

    # Always asks for a tool; the loop must force a final tool-less call.
    looping = LLMResponse(text="again",
                          tool_calls=[ToolCall("t", "search_10k", {"query": "q"})])

    client = ScriptedClient([looping, looping, LLMResponse(text="forced final")])

    final, _ = run_tool_loop(
        client, [{"role": "user", "text": "go"}], [tool], max_iterations=2,
    )

    assert final.text == "forced final"
    # Last call was made without tools.
    assert client.calls[-1]["tools"] is None


# --------------------------------------------------------------------------
# Transcript recorder integration
# --------------------------------------------------------------------------

from llm_client import LLMClient  # noqa: E402
from transcript import TranscriptRecorder  # noqa: E402
from settings import LLMConfig  # noqa: E402


class _RecordingStub:
    """Scripted client exposing a `recorder` attr, like the real LLMClient."""

    def __init__(self, responses, recorder=None):
        self.responses = responses
        self.calls = 0
        self.recorder = recorder

    def complete(self, messages, tools=None, system=None):
        resp = self.responses[self.calls]
        self.calls += 1
        return resp


def _search_tool():
    return Tool(
        name="search",
        description="search",
        parameters={"type": "object", "properties": {}},
        impl=lambda **kw: "TOOL-OUTPUT",
    )


def test_run_tool_loop_records_tool_calls():
    rec = TranscriptRecorder()
    call = ToolCall(id="c1", name="search", arguments={"query": "x"})
    client = _RecordingStub([
        LLMResponse(text="", tool_calls=[call], stop_reason="tool_use"),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn"),
    ], recorder=rec)

    run_tool_loop(client, [], [_search_tool()])

    # The scripted stand-in does not simulate llm_call recording (only the
    # real LLMClient.complete does), so only the tool execution is recorded.
    assert [e["type"] for e in rec.events] == ["tool_call"]
    assert rec.events[0]["name"] == "search"
    assert rec.events[0]["result"] == "TOOL-OUTPUT"


def test_run_tool_loop_without_recorder_is_unchanged():
    call = ToolCall(id="c1", name="search", arguments={"query": "x"})
    client = _RecordingStub([
        LLMResponse(text="", tool_calls=[call], stop_reason="tool_use"),
        LLMResponse(text="done", tool_calls=[], stop_reason="end_turn"),
    ])  # recorder defaults to None -> must not raise

    final, _ = run_tool_loop(client, [], [_search_tool()])
    assert final.text == "done"


def test_llmclient_complete_records_llm_call(monkeypatch):
    cfg = LLMConfig(
        backend="local", model="qwen",
        local_base_url="http://x/v1", local_api_key="k",
    )
    rec = TranscriptRecorder()
    client = LLMClient(cfg, recorder=rec)

    raw = {
        "choices": [{"message": {"content": "hi", "tool_calls": None},
                     "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }
    monkeypatch.setattr(client, "_raw_local", lambda request: raw)

    resp = client.complete([{"role": "user", "text": "hello"}])

    assert resp.text == "hi"
    assert len(rec.events) == 1
    assert rec.events[0]["type"] == "llm_call"
    assert rec.events[0]["backend"] == "local"
    assert rec.events[0]["usage"] == {"total_tokens": 5}
    assert rec.events[0]["raw_response"] == raw
