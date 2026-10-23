"""Generate small synthetic run directories matching contract C4.

Usage: python -m dagshare_exp.fixtures.make_fixtures [out_dir]
Default out_dir is dagshare_exp/fixtures/results.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

TENANTS = ("agent", "chat")
T0 = 1_760_000_000.0
DURATION_S = 60.0
WARMUP_S = 5.0
LEDGER_DT = 1.0
TOTAL_KV_GB = 20.0
MAX_RUNNING = 64

PRESETS = {
    "vllm": {"policy": "fcfs", "alpha": None, "beta": None, "fit": None},
    "continuum": {"policy": "continuum", "alpha": None, "beta": None, "fit": None},
    "A0": {"policy": "dagshare", "alpha": 0.0, "beta": 0.0, "fit": False},
    "A1": {"policy": "dagshare", "alpha": 0.0, "beta": 0.0, "fit": True},
    "A2": {"policy": "dagshare", "alpha": 1.0, "beta": 0.0, "fit": True},
    "A3": {"policy": "dagshare", "alpha": 1.0, "beta": 1.0, "fit": True},
}

# (preset, retention dir token, retention spec)
RUNS = [
    ("vllm", "none", "none"),
    ("continuum", "ttl2", "ttl:2"),
    *[(p, tok, spec) for p in ("A0", "A1", "A2", "A3")
      for tok, spec in (("ttl2", "ttl:2"), ("until_done60", "until_done:60"))],
]
SEEDS = (1, 2)

# Per-run behaviour knobs: (chat TTFT scale, agent turn slowdown, idle GB/pin, preempts)
BEHAVIOUR = {
    "vllm": (1.0, 1.6, 0.0, 6),
    "continuum": (2.2, 1.0, 1.5, 1),
    "A0": (1.9, 1.05, 1.4, 2),
    "A1": (1.6, 1.08, 1.3, 2),
    "A2": (1.4, 1.1, 1.2, 2),
    "A3": (1.1, 1.15, 0.8, 3),
}


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _request(tenant, job_id, turn, send_t, ttft, decode_s, prompt, output, status="ok",
             warmup=False) -> dict:
    done = status == "ok"
    return {
        "tenant": tenant, "job_id": job_id, "turn": turn,
        "send_t": round(send_t, 4),
        "first_token_t": round(send_t + ttft, 4) if done else None,
        "finish_t": round(send_t + ttft + decode_s, 4) if done else None,
        "prompt_tokens": prompt, "output_tokens": output if done else 0,
        "status": status, "warmup": warmup,
    }


def make_run(out: Path, preset: str, ret_tok: str, ret_spec: str, seed: int) -> None:
    rng = random.Random(f"{preset}-{ret_tok}-{seed}")
    chat_scale, agent_slow, idle_gb, n_preempt = BEHAVIOUR[preset]
    if ret_tok.startswith("until_done"):
        idle_gb *= 1.8
        chat_scale *= 1.15
    run_dir = out / f"{preset}_{ret_tok}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    reqs, events = [], []

    t = T0
    i = 0
    while t < T0 + DURATION_S:
        t += rng.expovariate(1.0)
        ttft = rng.uniform(0.05, 0.3) * chat_scale
        out_tok = rng.randint(50, 300)
        status = "ok" if rng.random() > 0.03 else "error"
        reqs.append(_request("chat", f"chat-{i}", 0, t, ttft, out_tok * 0.01,
                             rng.randint(100, 800), out_tok, status, t < T0 + WARMUP_S))
        i += 1

    rid = 0
    for slot in range(3):
        t = T0 + slot * 0.5
        j = 0
        while t < T0 + DURATION_S - 5:
            job_id = f"job-{slot}-{j}"
            n_turns = rng.randint(3, 5)
            warm = t < T0 + WARMUP_S
            for turn in range(n_turns):
                ttft = rng.uniform(0.1, 0.5) * agent_slow
                out_tok = rng.randint(80, 200)
                decode = out_tok * 0.012 * agent_slow
                reqs.append(_request("agent", job_id, turn, t, ttft, decode,
                                     2000 + 600 * turn, out_tok, warmup=warm))
                end = t + ttft + decode
                rid += 1
                is_last = turn == n_turns - 1
                tool_s = 0.0 if is_last else rng.uniform(0.5, 2.5)
                if preset != "vllm" and not is_last:
                    gb = round(0.1 + 0.03 * turn, 3)
                    events.append({"t": end, "event": "pin", "request_id": f"r{rid}",
                                   "job_id": job_id, "tenant_id": "agent", "gb": gb,
                                   "reason": "tool_call"})
                    events.append({"t": end + tool_s, "event": "unpin", "request_id": f"r{rid}",
                                   "job_id": job_id, "tenant_id": "agent", "gb": gb,
                                   "reason": "next_turn"})
                if is_last:
                    events.append({"t": end, "event": "release", "request_id": f"r{rid}",
                                   "job_id": job_id, "tenant_id": "agent", "gb": 0.1,
                                   "reason": "last_step"})
                t = end + tool_s
            j += 1
        if slot == 0:
            reqs.append(_request("agent", f"job-{slot}-{j}", 0, t, 0, 0, 2000, 0, "unfinished"))

    for k in range(n_preempt):
        tenant = "agent" if k % 2 == 0 else "chat"
        events.append({"t": T0 + 10 + 7 * k, "event": "preempt", "request_id": f"p{k}",
                       "job_id": None, "tenant_id": tenant, "gb": 0.2, "reason": "oom"})

    reqs.sort(key=lambda r: r["send_t"])
    events.sort(key=lambda e: e["t"])
    write_jsonl(run_dir / "requests.jsonl", reqs)

    info = PRESETS[preset]
    config = {"policy": info["policy"], "preset": preset, "retention": ret_spec, "seed": seed,
              "model": "meta-llama/Llama-3.1-8B-Instruct", "git_commit": "fixture",
              "env": {}}
    if info["policy"] == "dagshare":
        alpha, beta = info["alpha"], info["beta"]
        config["env"] = {"DAGSHARE_ALPHA": str(alpha), "DAGSHARE_BETA": str(beta),
                         "DAGSHARE_FIT_CHECK": "on" if info["fit"] else "off",
                         "DAGSHARE_COMPUTE": "on", "KV_RETENTION": ret_spec}
        cum = {tn: {"C": 0.0, "M_act": 0.0, "M_idle": 0.0} for tn in TENANTS}
        ledger = []
        tt = T0
        while tt <= T0 + DURATION_S:
            gb_run = {"agent": rng.uniform(4, 8), "chat": rng.uniform(1, 3)}
            gb_pin = {"agent": idle_gb * rng.uniform(0.5, 1.5), "chat": 0.0}
            slots = {"agent": rng.randint(2, 4), "chat": rng.randint(1, 3)}
            for tn in TENANTS:
                cum[tn]["C"] += slots[tn] * LEDGER_DT
                cum[tn]["M_act"] += gb_run[tn] * LEDGER_DT
                cum[tn]["M_idle"] += gb_pin[tn] * LEDGER_DT
            tenants = {}
            for tn in TENANTS:
                c = cum[tn]
                s = (c["C"] / MAX_RUNNING + alpha * c["M_act"] / TOTAL_KV_GB
                     + beta * c["M_idle"] / TOTAL_KV_GB)
                tenants[tn] = {k: round(v, 4) for k, v in c.items()} | {"S": round(s, 4)}
            ledger.append({"t": round(tt, 3), "tenants": tenants,
                           "gb_running": round(sum(gb_run.values()), 3),
                           "gb_pinned": round(sum(gb_pin.values()), 3),
                           "n_running": sum(slots.values()), "n_waiting": rng.randint(0, 5),
                           "n_pinned": rng.randint(0, 3)})
            tt += LEDGER_DT
        write_jsonl(run_dir / "ledger.jsonl", ledger)
        write_jsonl(run_dir / "events.jsonl", events)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for preset, tok, spec in RUNS:
        for seed in SEEDS:
            make_run(out, preset, tok, spec, seed)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results")
