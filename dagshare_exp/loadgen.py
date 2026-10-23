"""DAGShare load generator: agent (closed-loop trace replay) and chat (open-loop Poisson) tenants.

Usage:
  python -m dagshare_exp.loadgen --workload dagshare_exp/configs/workloads/smoke.yaml \
      --base-url http://localhost:8000 --seed 1 --out results/run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import yaml

EXP_DIR = Path(__file__).resolve().parent
AGENT_DEFAULTS = {"jobs_in_flight": 8, "tool_time_scale": 1.0, "ramp_s": 0.0}
CHAT_DEFAULTS = {"rate_rps": 1.0, "max_output_tokens": None}
TOKEN_ID_RANGE = (1000, 30000)
CONTEXT_MARGIN = 8


@dataclass
class TenantSpec:
    id: str
    kind: str
    cfg: dict[str, Any]


@dataclass
class Workload:
    model: str | None
    max_model_len: int
    duration_s: float
    warmup_s: float
    tenants: list[TenantSpec]
    raw: dict[str, Any] = field(default_factory=dict)


def resolve_path(p: str, workload_path: Path) -> Path:
    """Resolve a data path: absolute, cwd-relative, or relative to dagshare_exp/."""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    for base in (workload_path.parent.parent.parent, EXP_DIR):
        if (base / path).exists():
            return base / path
    return EXP_DIR / path


def load_workload(path: str | Path) -> Workload:
    """Parse a C5 workload YAML; top-level `agent:`/`chat:` blocks are per-kind defaults for tenants."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    tenants = []
    for t in raw.get("tenants") or []:
        kind = t["kind"]
        if kind not in ("agent", "chat"):
            raise ValueError(f"tenant {t.get('id')}: unknown kind {kind!r}")
        defaults = AGENT_DEFAULTS if kind == "agent" else CHAT_DEFAULTS
        cfg = {**defaults, **(raw.get(kind) or {}), **t}
        key = "traces" if kind == "agent" else "source"
        if key not in cfg:
            raise ValueError(f"tenant {t['id']}: missing {key!r}")
        cfg[key] = str(resolve_path(cfg[key], path))
        tenants.append(TenantSpec(id=str(t["id"]), kind=kind, cfg=cfg))
    if not tenants:
        raise ValueError(f"{path}: no tenants")
    if len({t.id for t in tenants}) != len(tenants):
        raise ValueError(f"{path}: duplicate tenant ids")
    return Workload(
        model=raw.get("model"),
        max_model_len=int(raw.get("max_model_len", 32768)),
        duration_s=float(raw.get("duration_s", 60)),
        warmup_s=float(raw.get("warmup_s", 0)),
        tenants=tenants,
        raw=raw,
    )


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def random_tokens(rng: random.Random, n: int) -> list[int]:
    lo, hi = TOKEN_ID_RANGE
    return [rng.randrange(lo, hi) for _ in range(max(0, n))]


class Recorder:
    """Appends one C4 row per request to requests.jsonl and keeps counters."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "requests.jsonl"
        self._f = open(self.path, "w")
        self.rows = 0
        self.by_status: dict[str, int] = {}
        self.by_tenant: dict[str, int] = {}
        self.warmup = 0
        self.jobs_done: dict[str, int] = {}

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()
        self.rows += 1
        self.by_status[row["status"]] = self.by_status.get(row["status"], 0) + 1
        self.by_tenant[row["tenant"]] = self.by_tenant.get(row["tenant"], 0) + 1
        self.warmup += bool(row["warmup"])

    def close(self) -> None:
        self._f.close()


class Runner:
    def __init__(self, args: argparse.Namespace, workload: Workload):
        self.args = args
        self.wl = workload
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model or workload.model
        self.seed = args.seed
        self.duration_s = args.duration_s if args.duration_s is not None else workload.duration_s
        self.warmup_s = args.warmup_s if args.warmup_s is not None else workload.warmup_s
        self.rec = Recorder(Path(args.out))
        self.t0 = 0.0
        self.stop_at = 0.0
        self.warmup_until = 0.0
        self.stopping = asyncio.Event()
        self.inflight: set[asyncio.Task] = set()
        self.session: aiohttp.ClientSession | None = None

    def is_stopping(self) -> bool:
        return self.stopping.is_set() or time.time() >= self.stop_at

    async def sleep_or_stop(self, seconds: float) -> bool:
        """Sleep up to `seconds`; return True if the run is stopping."""
        deadline = min(time.time() + max(0.0, seconds), self.stop_at)
        try:
            await asyncio.wait_for(self.stopping.wait(), timeout=max(0.0, deadline - time.time()))
        except asyncio.TimeoutError:
            pass
        return self.is_stopping()

    async def send(self, *, tenant: str, kind: str, job_id: str, turn: int, prompt: list[int] | str,
                   prompt_tokens_est: int, max_tokens: int, is_last_step: bool,
                   this_func_call: str | None, last_func_call: str | None) -> tuple[dict, list[int]]:
        """Send one streaming /v1/completions request; returns the C4 row and generated token ids."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "tenant_id": tenant,
            "job_id": job_id,
            "is_last_step": is_last_step,
            "this_func_call": this_func_call,
            "last_func_call": last_func_call,
        }
        send_t = time.time()
        row = {
            "tenant": tenant, "kind": kind, "job_id": job_id, "turn": turn,
            "send_t": send_t, "first_token_t": None, "finish_t": None,
            "prompt_tokens": prompt_tokens_est, "output_tokens": 0,
            "status": "ok", "warmup": send_t < self.warmup_until,
            "is_last_step": is_last_step, "this_func_call": this_func_call,
        }
        out_ids: list[int] = []
        n_text_chunks = 0
        try:
            async with self.session.post(f"{self.base_url}/v1/completions", json=body) as resp:
                if resp.status != 200:
                    row["status"] = "error"
                    row["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                else:
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            row["status"] = "error"
                            row["error"] = str(chunk["error"])[:300]
                            break
                        for choice in chunk.get("choices") or []:
                            ids = choice.get("token_ids") or []
                            if ids or choice.get("text"):
                                if row["first_token_t"] is None:
                                    row["first_token_t"] = time.time()
                                out_ids.extend(ids)
                                n_text_chunks += 1
                        usage = chunk.get("usage")
                        if usage:
                            row["prompt_tokens"] = usage.get("prompt_tokens", row["prompt_tokens"])
                            row["output_tokens"] = usage.get("completion_tokens", row["output_tokens"])
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            row["finish_t"] = time.time()
            row["output_tokens"] = row["output_tokens"] or len(out_ids) or n_text_chunks
            self.rec.write(row)
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        row["finish_t"] = time.time()
        row["output_tokens"] = row["output_tokens"] or len(out_ids) or n_text_chunks
        self.rec.write(row)
        return row, out_ids

    async def tracked(self, coro):
        """Run a request as a tracked task so the drain phase can cancel it."""
        task = asyncio.ensure_future(coro)
        self.inflight.add(task)
        try:
            return await task
        finally:
            self.inflight.discard(task)

    async def agent_job(self, spec: TenantSpec, job_seq: int, trace: dict, prefix: list[int]) -> bool:
        """Replay one trace job; returns True if every turn completed."""
        rng = random.Random(f"{self.seed}:{spec.id}:job:{job_seq}")
        job_id = f"{spec.id}-{job_seq}-{trace.get('job_id', job_seq)}"
        turns = trace["turns"]
        history: list[int] = []
        last_tool: str | None = None
        scale = float(spec.cfg["tool_time_scale"])
        for i, turn in enumerate(turns):
            if self.is_stopping():
                return False
            is_last = i == len(turns) - 1
            target = int(turn["prompt_tokens"])
            if not history:
                history = prefix + random_tokens(rng, target - len(prefix))
            elif len(history) < target:
                history += random_tokens(rng, target - len(history))
            max_tokens = max(1, int(turn["output_tokens"]))
            if len(history) + max_tokens > self.wl.max_model_len - CONTEXT_MARGIN:
                return False
            tool = None if is_last else turn.get("tool_name")
            row, out_ids = await self.tracked(self.send(
                tenant=spec.id, kind="agent", job_id=job_id, turn=i, prompt=list(history),
                prompt_tokens_est=len(history), max_tokens=max_tokens, is_last_step=is_last,
                this_func_call=tool, last_func_call=last_tool))
            if row["status"] != "ok":
                return False
            if is_last:
                return True
            if len(out_ids) < max_tokens:
                out_ids = out_ids + random_tokens(rng, max_tokens - len(out_ids))
            history += out_ids[:max_tokens]
            if await self.sleep_or_stop(float(turn.get("tool_time_s", 0.0)) * scale):
                return False
            history += random_tokens(rng, int(turn.get("tool_output_tokens", 0)))
            last_tool = tool
        return True

    async def agent_slot(self, spec: TenantSpec, slot: int, traces: list[dict], order: list[int],
                         counter: list[int], prefix: list[int]) -> None:
        ramp = float(spec.cfg.get("ramp_s") or 0.0)
        n_slots = int(spec.cfg["jobs_in_flight"])
        if ramp > 0 and await self.sleep_or_stop(ramp * slot / n_slots):
            return
        while not self.is_stopping():
            job_seq = counter[0]
            counter[0] += 1
            trace = traces[order[job_seq % len(order)]]
            if await self.agent_job(spec, job_seq, trace, prefix):
                self.rec.jobs_done[spec.id] = self.rec.jobs_done.get(spec.id, 0) + 1

    async def agent_tenant(self, spec: TenantSpec) -> None:
        traces = read_jsonl(spec.cfg["traces"])
        if not traces:
            raise ValueError(f"{spec.cfg['traces']}: empty trace file")
        rng = random.Random(f"{self.seed}:{spec.id}:order")
        order = list(range(len(traces)))
        rng.shuffle(order)
        prefix = random_tokens(rng, int(spec.cfg.get("shared_prefix_tokens", 0)))
        counter = [0]
        await asyncio.gather(*(self.agent_slot(spec, s, traces, order, counter, prefix)
                               for s in range(int(spec.cfg["jobs_in_flight"]))))

    async def chat_tenant(self, spec: TenantSpec) -> None:
        prompts = read_jsonl(spec.cfg["source"])
        cap = spec.cfg.get("max_output_tokens")
        limit = self.wl.max_model_len - CONTEXT_MARGIN
        prompts = [p for p in prompts if int(p.get("prompt_tokens", len(p["prompt"]) // 4))
                   + min(int(p["output_tokens"]), int(cap or 1 << 30)) <= limit]
        if not prompts:
            raise ValueError(f"{spec.cfg['source']}: no usable chat prompts")
        rng = random.Random(f"{self.seed}:{spec.id}:chat")
        rate = float(spec.cfg["rate_rps"])
        tasks: list[asyncio.Task] = []
        n = 0
        next_t = time.time()
        while True:
            next_t += rng.expovariate(rate)
            p = prompts[rng.randrange(len(prompts))]
            if next_t >= self.stop_at or await self.sleep_or_stop(next_t - time.time()):
                break
            max_tokens = max(1, int(p["output_tokens"]))
            if cap:
                max_tokens = min(max_tokens, int(cap))
            tasks.append(asyncio.ensure_future(self.tracked(self.send(
                tenant=spec.id, kind="chat", job_id=f"{spec.id}-r{n}", turn=0, prompt=p["prompt"],
                prompt_tokens_est=int(p.get("prompt_tokens", len(p["prompt"]) // 4)),
                max_tokens=max_tokens, is_last_step=True, this_func_call=None, last_func_call=None))))
            n += 1
        await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_health(self) -> None:
        deadline = time.time() + self.args.wait_health_s
        while True:
            try:
                async with self.session.get(f"{self.base_url}/health") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            if time.time() >= deadline:
                raise RuntimeError(f"{self.base_url}/health not ready after {self.args.wait_health_s}s")
            await asyncio.sleep(1.0)

    async def run(self) -> dict:
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.args.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
            self.session = s
            if self.args.wait_health_s > 0:
                await self.wait_health()
            if not self.model:
                async with s.get(f"{self.base_url}/v1/models") as r:
                    self.model = (await r.json())["data"][0]["id"]
            self.t0 = time.time()
            self.warmup_until = self.t0 + self.warmup_s
            self.stop_at = self.warmup_until + self.duration_s
            loops = [self.agent_tenant(t) if t.kind == "agent" else self.chat_tenant(t)
                     for t in self.wl.tenants]
            main = asyncio.ensure_future(asyncio.gather(*loops))
            try:
                await asyncio.wait_for(asyncio.shield(main), timeout=self.stop_at - time.time()
                                       + self.args.drain_s)
            except asyncio.TimeoutError:
                self.stopping.set()
                for task in list(self.inflight):
                    task.cancel()
                await asyncio.gather(main, return_exceptions=True)
            self.stopping.set()
        self.rec.close()
        return self.summary()

    def summary(self) -> dict:
        return {
            "requests": self.rec.rows, "status": self.rec.by_status, "per_tenant": self.rec.by_tenant,
            "warmup_requests": self.rec.warmup, "agent_jobs_completed": self.rec.jobs_done,
            "wall_s": round(time.time() - self.t0, 2), "warmup_s": self.warmup_s,
            "duration_s": self.duration_s, "seed": self.seed, "out": str(self.rec.path),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", required=True, help="workload YAML (C5)")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="served model name (default: workload model, else /v1/models)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True, help="output dir for requests.jsonl and loadgen.json")
    p.add_argument("--duration-s", type=float, default=None, help="measured period (overrides workload)")
    p.add_argument("--warmup-s", type=float, default=None, help="warmup period before it (overrides workload)")
    p.add_argument("--drain-s", type=float, default=30.0, help="grace for in-flight requests after the end")
    p.add_argument("--wait-health-s", type=float, default=600.0, help="wait for /health first (0 = skip)")
    p.add_argument("--request-timeout-s", type=float, default=600.0, help="max gap between stream chunks")
    p.add_argument("--done-file", default=None, help="touch this file when finished (e.g. /run/done)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workload = load_workload(args.workload)
    runner = Runner(args, workload)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "workload": workload.raw, "model": runner.model,
            "tenants": [{"id": t.id, "kind": t.kind, **t.cfg} for t in workload.tenants],
            "start_t": time.time()}
    (out / "loadgen.json").write_text(json.dumps(meta, indent=2, default=str))
    summary = asyncio.run(runner.run())
    print("LOADGEN_SUMMARY " + json.dumps(summary), flush=True)
    if args.done_file:
        Path(args.done_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.done_file).touch()
    return 0 if summary["requests"] and summary["status"].get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
