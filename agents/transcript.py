"""
Transcript logging for the agent layer.

The LLM client and tool loop are otherwise opaque: the system prompt, each
model turn, every tool call's arguments and returned content, token usage,
and the raw provider payloads are computed and then discarded. A
`TranscriptRecorder` captures all of that as an ordered event stream so a run
can be inspected, debugged, and later reused (e.g. for ablation).

Capture is built into `LLMClient`/`run_tool_loop` (see llm_client.py): pass a
recorder to `LLMClient(config, recorder=...)` and it records every model call;
`run_tool_loop` records every tool execution against the same recorder, so the
events interleave in true chronological order:

    llm_call -> tool_call(s) -> llm_call -> ... -> final llm_call

The recorder is the source of truth for the conversation, not the `messages`
list: it captures even the final no-tool answer and the cap-reached forced
call, which the loop does not append to `messages`.

Persisted as `transcript.json` (structured) plus `transcript.md` (human
readable). Raw provider payloads are recorded by default; set
`capture_raw=False` to keep only the normalized conversation + token usage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path


def _extract_usage(raw_response):
    """
    Pull a token-usage dict from a raw provider response, or None.

    Anthropic returns {"usage": {"input_tokens", "output_tokens", ...}};
    OpenAI-compatible servers return {"usage": {"prompt_tokens",
    "completion_tokens", "total_tokens"}}. Both are passed through as-is.
    """
    if not isinstance(raw_response, dict):
        return None

    return raw_response.get("usage")


def _jsonable(value):
    """
    Best-effort conversion of dataclasses (ToolCall/ToolResult) and other
    objects into JSON-serializable structures.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class TranscriptRecorder:
    """
    Collects an ordered list of `llm_call` and `tool_call` events.
    """

    def __init__(self, metadata=None, capture_raw=True):
        self.metadata = dict(metadata or {})
        self.capture_raw = capture_raw
        self.events: list[dict] = []

    def record_llm_call(self, backend, model, request, raw_response, response):
        """
        Record one model call: the normalized assistant output (text +
        tool calls), stop reason, and token usage. When capture_raw is set,
        also the exact request dict and raw provider response.
        """
        event = {
            "type": "llm_call",
            "backend": backend,
            "model": model,
            "assistant_text": response.text,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in response.tool_calls
            ],
            "stop_reason": response.stop_reason,
            "usage": _extract_usage(raw_response),
        }

        if self.capture_raw:
            event["raw_request"] = _jsonable(request)
            event["raw_response"] = _jsonable(raw_response)

        self.events.append(event)

    def record_tool_call(self, name, arguments, result, error=None):
        """
        Record one tool execution: the arguments the model passed and the
        content returned to it (or an error string).
        """
        self.events.append({
            "type": "tool_call",
            "name": name,
            "arguments": _jsonable(arguments),
            "result": result,
            "error": error,
        })

    def to_dict(self):
        return {"metadata": self.metadata, "events": self.events}

    def to_markdown(self):
        """
        Render the transcript as human-readable Markdown: metadata, the
        system prompt (if present), the initial user prompt (if available),
        then each event in order.
        """
        lines: list[str] = ["# Agent transcript", ""]

        if self.metadata:
            lines.append("## Metadata")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        system = self._system_prompt()
        if system:
            lines.append("## System prompt")
            lines.append("")
            lines.append("```")
            lines.append(system)
            lines.append("```")
            lines.append("")

        initial_prompt = self._initial_user_prompt()
        if initial_prompt:
            lines.append("## Initial user prompt")
            lines.append("")
            lines.append("```")
            lines.append(initial_prompt)
            lines.append("```")
            lines.append("")

        lines.append("## Conversation")
        lines.append("")

        step = 0
        for event in self.events:
            if event["type"] == "llm_call":
                step += 1
                lines.append(f"### Step {step} — model ({event.get('model')})")
                usage = event.get("usage")
                if usage:
                    lines.append(f"_usage: {usage}_")
                lines.append("")
                if event.get("assistant_text"):
                    lines.append(event["assistant_text"])
                    lines.append("")
                for call in event.get("tool_calls", []):
                    args = json.dumps(call["arguments"], ensure_ascii=False)
                    lines.append(f"**→ tool call** `{call['name']}` {args}")
                    lines.append("")
            else:  # tool_call
                label = "tool error" if event.get("error") else "tool result"
                lines.append(f"**← {label}** `{event['name']}`")
                lines.append("")
                lines.append("```")
                lines.append(str(event.get("error") or event.get("result") or ""))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _system_prompt(self):
        """
        Recover the system prompt from the first recorded raw request, if
        raw capture was on. Anthropic puts it at request["system"]; OpenAI
        puts a {"role": "system"} message first in request["messages"].
        """
        for event in self.events:
            if event["type"] != "llm_call":
                continue
            request = event.get("raw_request")
            if not isinstance(request, dict):
                continue
            if request.get("system"):
                return request["system"]
            for msg in request.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "system":
                    return msg.get("content")
        return None

    def _initial_user_prompt(self):
        """
        Recover the first user message from the first recorded raw request,
        so the rendered transcript can show the actual prompt the model saw.
        """
        for event in self.events:
            if event["type"] != "llm_call":
                continue
            request = event.get("raw_request")
            if not isinstance(request, dict):
                continue

            messages = request.get("messages", [])
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user":
                    return msg.get("content")

        return None

    def save(self, run_dir):
        """
        Atomically write transcript.json and transcript.md into run_dir.

        Returns (json_path, md_path).
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        json_path = run_dir / "transcript.json"
        md_path = run_dir / "transcript.md"

        _atomic_write(
            json_path,
            json.dumps(self.to_dict(), indent=2, default=str),
        )
        _atomic_write(md_path, self.to_markdown())

        return json_path, md_path


def new_run_dir(base_dir, label):
    """
    Create and return base_dir/<label>_<YYYYmmdd_HHMMSS> for one run's
    artifacts.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _atomic_write(path, content):
    """
    Write content to path via a temporary file + rename, so partial writes
    from interrupted runs do not corrupt the artifact.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    tmp_path.replace(path)
