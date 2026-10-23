"""Record SWE-bench runs with mini-swe-agent and convert its trajectories to agent traces (C3).

  record:  python dagshare_exp/traces/convert_mini_swe.py record -- swebench --model-class vllm ...
  convert: python dagshare_exp/traces/convert_mini_swe.py convert RUN_DIR --out dagshare_exp/traces/swebench.jsonl

`record` runs `mini-extra` with hooks that store per-step tool wall time (`tool_time_s` on the
observation message) and request token usage (`extra.response.usage` on assistant messages).
`convert` also works on plain trajectories, falling back to estimated tokens and default tool times.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

BASH_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
MSG_OVERHEAD_TOKENS = 4
DEFAULT_TOOL_TIME_S = 1.0


def install_record_hooks() -> None:
    """Patch mini-swe-agent to record tool wall time and token usage into trajectories."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.vllm_model import VllmModel

    orig_filter = VllmModel._filter_openai_params

    def filter_params(params):
        out = orig_filter(params)
        if params.get("stream", True):
            out["stream_options"] = {"include_usage": True}
        return out

    VllmModel._filter_openai_params = staticmethod(filter_params)

    orig_execute = DefaultAgent.execute_action
    orig_add = DefaultAgent.add_message

    def execute_action(self, action):
        t = time.time()
        try:
            return orig_execute(self, action)
        finally:
            self._dagshare_tool_time = time.time() - t

    def add_message(self, role, content, **kwargs):
        pending = getattr(self, "_dagshare_tool_time", None)
        if role == "user" and pending is not None:
            kwargs["tool_time_s"] = round(pending, 4)
            self._dagshare_tool_time = None
        if role == "assistant":
            kwargs["t_done"] = time.time()
        return orig_add(self, role, content, **kwargs)

    DefaultAgent.execute_action = execute_action
    DefaultAgent.add_message = add_message


def make_counter(tokenizer_name: str | None):
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda text: len(tok(text, add_special_tokens=False).input_ids)
    return lambda text: max(1, math.ceil(len(text) / 4))


def tool_of(content: str) -> str | None:
    """First word of the single bash block, as Continuum's ToolCallParser does."""
    actions = BASH_RE.findall(content or "")
    if len(actions) == 1 and actions[0].strip():
        return actions[0].strip().split()[0]
    return None


def usage_of(msg: dict) -> dict | None:
    resp = ((msg.get("extra") or {}).get("response")) or {}
    usage = resp.get("usage")
    return usage if usage and usage.get("prompt_tokens") else None


def convert_messages(job_id: str, messages: list[dict], count, default_tool_time: dict[str, float]) -> dict | None:
    """Turn one trajectory's message list into a C3 job, or None if it has no model calls."""
    steps = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not steps:
        return None
    msg_tokens = [count(str(m.get("content", ""))) + MSG_OVERHEAD_TOKENS for m in messages]
    turns = []
    for k, i in enumerate(steps):
        msg = messages[i]
        usage = usage_of(msg)
        prompt = usage["prompt_tokens"] if usage else sum(msg_tokens[:i])
        output = usage.get("completion_tokens") if usage else None
        output = output or count(str(msg.get("content", "")))
        nxt = steps[k + 1] if k + 1 < len(steps) else len(messages)
        observations = messages[i + 1:nxt]
        turns.append({"prompt_tokens": int(prompt), "output_tokens": int(output),
                      "tool_name": tool_of(str(msg.get("content", ""))),
                      "tool_time_s": next((float(o["tool_time_s"]) for o in observations if "tool_time_s" in o), None),
                      "tool_output_tokens": sum(msg_tokens[i + 1:nxt])})
    for k in range(len(turns) - 1):
        t, n = turns[k], turns[k + 1]
        if usage_of(messages[steps[k + 1]]):
            t["tool_output_tokens"] = max(0, n["prompt_tokens"] - t["prompt_tokens"] - t["output_tokens"])
        if t["tool_time_s"] is None:
            t["tool_time_s"] = default_tool_time.get(t["tool_name"] or "", default_tool_time.get("*", DEFAULT_TOOL_TIME_S))
            if t["tool_name"] is None:
                t["tool_time_s"] = 0.0
        t["tool_time_s"] = round(t["tool_time_s"], 4)
    turns[-1].update(tool_name=None, tool_time_s=0.0, tool_output_tokens=0)
    return {"job_id": job_id, "source": "swebench", "turns": turns}


def find_trajs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.traj.json")) if p.is_dir() else [p]
    return out


def cmd_convert(a: argparse.Namespace) -> None:
    count = make_counter(a.tokenizer)
    defaults = json.loads(a.default_tool_time) if a.default_tool_time else {}
    jobs, skipped, timed = [], 0, 0
    for path in find_trajs(a.inputs):
        data = json.loads(path.read_text())
        job_id = data.get("instance_id") or path.name.removesuffix(".traj.json")
        job = convert_messages(job_id, data.get("messages") or [], count, defaults)
        if job is None or len(job["turns"]) < a.min_turns:
            skipped += 1
            continue
        timed += any("tool_time_s" in m for m in data["messages"])
        jobs.append(job)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(jobs)} jobs to {a.out} (skipped {skipped}; {timed} with recorded tool times)")


def cmd_record(a: argparse.Namespace) -> None:
    install_record_hooks()
    from minisweagent.run.mini_extra import main as mini_extra_main
    sys.argv = ["mini-extra", *a.mini_args]
    mini_extra_main()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="trajectories -> C3 JSONL")
    c.add_argument("inputs", nargs="+", help="mini-swe-agent output dirs or .traj.json files")
    c.add_argument("--out", default=str(Path(__file__).resolve().parent / "swebench.jsonl"))
    c.add_argument("--tokenizer", default=None, help="HF tokenizer for estimates when usage is missing")
    c.add_argument("--default-tool-time", default=None,
                   help='JSON {tool: seconds, "*": seconds} used when tool time was not recorded')
    c.add_argument("--min-turns", type=int, default=2)
    r = sub.add_parser("record", help="run mini-extra with recording hooks; args after -- go to mini-extra")
    r.add_argument("mini_args", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.cmd == "record":
        a.mini_args = a.mini_args[1:] if a.mini_args[:1] == ["--"] else a.mini_args
        cmd_record(a)
    else:
        cmd_convert(a)


if __name__ == "__main__":
    main()
