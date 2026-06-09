"""
Unified LLM client for the agent layer.

Supports two backends behind one interface (selected by config.LLMConfig):

    - "anthropic" : the hosted Anthropic Messages API.
    - "local"     : any OpenAI-compatible server (this project uses vLLM).

Both expose tool-calling, but their request/response shapes differ. The
strategy here mirrors the rest of the repo:

    - The format conversion and response parsing are PURE functions
      (to_anthropic_*/to_openai_*/parse_*), unit-tested with plain dicts.
    - The actual network call is a thin boundary (`_raw_anthropic` /
      `_raw_local`) that imports the SDK and returns `.model_dump()`, so
      the parsers always see plain dicts.
    - The tool-use loop (`run_tool_loop`) is tested by monkeypatching
      `LLMClient.complete` with scripted responses - no SDK needed.

Internal message format (backend-agnostic), a list of dicts:

    {"role": "user",      "text": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_results": [ToolResult, ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from settings import LLMConfig


DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.2


# --------------------------------------------------------------------------
# Normalized types
# --------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    id: str
    content: str


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass
class Tool:
    """A tool the model can call: schema + the Python implementation."""

    name: str
    description: str
    parameters: dict                     # JSON schema for the arguments
    impl: Callable[..., str]             # called with **arguments, returns str


# --------------------------------------------------------------------------
# Tool-definition conversion (pure)
# --------------------------------------------------------------------------

def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


# --------------------------------------------------------------------------
# Message conversion (pure)
# --------------------------------------------------------------------------

def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []

    for message in messages:
        role = message["role"]

        if role == "user":
            out.append({"role": "user", "content": message["text"]})

        elif role == "assistant":
            content: list[dict] = []

            if message.get("text"):
                content.append({"type": "text", "text": message["text"]})

            for call in message.get("tool_calls", []):
                content.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                })

            out.append({"role": "assistant", "content": content})

        elif role == "tool":
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.content,
                }
                for result in message["tool_results"]
            ]
            # Anthropic carries tool results in a user-role message.
            out.append({"role": "user", "content": content})

        else:
            raise ValueError(f"Unknown message role: {role!r}")

    return out


def to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    out: list[dict] = []

    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = message["role"]

        if role == "user":
            out.append({"role": "user", "content": message["text"]})

        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": message.get("text") or None}

            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ]

            out.append(msg)

        elif role == "tool":
            # OpenAI carries one message per tool result.
            for result in message["tool_results"]:
                out.append({
                    "role": "tool",
                    "tool_call_id": result.id,
                    "content": result.content,
                })

        else:
            raise ValueError(f"Unknown message role: {role!r}")

    return out


# --------------------------------------------------------------------------
# Request building (pure)
# --------------------------------------------------------------------------

def build_anthropic_request(
    model, messages, tools, system, max_tokens, temperature,
) -> dict:
    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": to_anthropic_messages(messages),
    }

    if system:
        request["system"] = system

    if tools:
        request["tools"] = to_anthropic_tools(tools)

    return request


def build_openai_request(
    model, messages, tools, system, max_tokens, temperature,
) -> dict:
    request: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": to_openai_messages(messages, system),
    }

    if tools:
        request["tools"] = to_openai_tools(tools)

    return request


def _adapt_openai_request(request: dict, exc) -> dict | None:
    """
    Rewrite an OpenAI Chat Completions request for a newer model that rejected
    a classic parameter, based on the 400 error. Returns an adapted copy, or
    None when the error is not a known adaptable-parameter case (so the caller
    re-raises). Pure: does not touch the network or mutate `request`.

    Handles:
      - `max_tokens` -> `max_completion_tokens` (required by o-series / gpt-5).
      - dropping `temperature` when the model only allows its default.
    """
    param = getattr(exc, "param", None)
    message = str(getattr(exc, "message", "") or exc)
    new = dict(request)

    if (param == "max_tokens" or
            ("max_tokens" in message and "max_completion_tokens" in message)):
        if "max_tokens" in new:
            new["max_completion_tokens"] = new.pop("max_tokens")
            return new

    if (param == "temperature" or
            ("temperature" in message and "support" in message.lower())):
        if "temperature" in new:
            new.pop("temperature")
            return new

    return None


# --------------------------------------------------------------------------
# Response parsing (pure)
# --------------------------------------------------------------------------

def parse_anthropic_response(raw: dict) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in raw.get("content", []) or []:
        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            tool_calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=block.get("input", {}) or {},
            ))

    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=raw.get("stop_reason"),
    )


def parse_openai_response(raw: dict) -> LLMResponse:
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}

    tool_calls: list[ToolCall] = []

    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) or {}
        arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}

        tool_calls.append(ToolCall(
            id=call.get("id", ""),
            name=function.get("name", ""),
            arguments=arguments or {},
        ))

    return LLMResponse(
        text=message.get("content") or "",
        tool_calls=tool_calls,
        stop_reason=choice.get("finish_reason"),
    )


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class LLMClient:
    """
    Backend-agnostic chat client with tool-calling.

    `complete` takes the internal message list, optional normalized tool
    defs ({name, description, parameters}), and an optional system prompt,
    and returns a normalized LLMResponse.
    """

    def __init__(
        self,
        config: LLMConfig,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        recorder=None,
    ):
        self.config = config
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Optional TranscriptRecorder (transcript.py). When set, every model
        # call is recorded with its raw payloads + token usage.
        self.recorder = recorder

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        if self.config.backend == "anthropic":
            request = build_anthropic_request(
                self.config.model, messages, tools, system,
                self.max_tokens, self.temperature,
            )
            raw = self._raw_anthropic(request)
            response = parse_anthropic_response(raw)
            backend = "anthropic"
        else:
            request = build_openai_request(
                self.config.model, messages, tools, system,
                self.max_tokens, self.temperature,
            )
            raw = self._raw_local(request)
            response = parse_openai_response(raw)
            backend = "local"

        if self.recorder is not None:
            self.recorder.record_llm_call(
                backend, self.config.model, request, raw, response
            )

        return response

    # -- network boundaries (imports happen here; not exercised offline) ----

    def _raw_anthropic(self, request: dict) -> dict:
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.require_api_key())
        message = client.messages.create(**request)
        return message.model_dump()

    def _raw_local(self, request: dict) -> dict:
        import openai

        client = openai.OpenAI(
            base_url=self.config.require_base_url(),
            api_key=self.config.require_api_key(),
        )

        # Newer OpenAI models (o-series, gpt-5, ...) reject the classic
        # Chat Completions params: `max_tokens` must be `max_completion_tokens`,
        # and they only allow the default `temperature`. Older models and local
        # vLLM/ollama servers want the classic params, so we send those by
        # default and only adapt on a 400 that names an unsupported param,
        # retrying until the request is accepted (or the error is unrelated).
        request = dict(request)
        for _ in range(4):
            try:
                return client.chat.completions.create(**request).model_dump()
            except openai.BadRequestError as exc:
                adapted = _adapt_openai_request(request, exc)
                if adapted is None or adapted == request:
                    raise
                request = adapted

        return client.chat.completions.create(**request).model_dump()


# --------------------------------------------------------------------------
# Tool-use loop
# --------------------------------------------------------------------------

def run_tool_loop(
    client: LLMClient,
    messages: list[dict],
    tools: list[Tool],
    system: str | None = None,
    max_iterations: int = 8,
) -> tuple[LLMResponse, list[dict]]:
    """
    Drive a multi-turn tool-use conversation until the model answers
    without calling a tool, or the iteration cap is hit.

    On each turn: call the model; if it requested tools, execute each
    implementation, append the assistant turn and the tool results, and
    loop. A failing or unknown tool returns an error string to the model
    rather than raising, so one bad call cannot crash the run.

    If the cap is reached while the model is still calling tools, one
    final call is made WITHOUT tools to force a textual answer. `messages`
    is mutated in place and also returned alongside the final response.
    """
    tool_defs = [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in tools
    ]
    impls = {t.name: t.impl for t in tools}

    for _ in range(max_iterations):
        response = client.complete(messages, tools=tool_defs, system=system)

        if not response.tool_calls:
            return response, messages

        messages.append({
            "role": "assistant",
            "text": response.text,
            "tool_calls": response.tool_calls,
        })

        results: list[ToolResult] = []

        for call in response.tool_calls:
            impl = impls.get(call.name)
            error = None

            if impl is None:
                content = f"Error: unknown tool {call.name!r}."
                error = content
            else:
                try:
                    content = impl(**call.arguments)
                except Exception as exc:  # noqa: BLE001 - feed error back to model
                    content = f"Error executing {call.name!r}: {exc}"
                    error = content

            recorder = getattr(client, "recorder", None)
            if recorder is not None:
                recorder.record_tool_call(
                    call.name, call.arguments, content, error=error
                )

            results.append(ToolResult(id=call.id, content=content))

        messages.append({"role": "tool", "tool_results": results})

    # Cap reached while still calling tools: force a final answer.
    final = client.complete(messages, tools=None, system=system)
    return final, messages
