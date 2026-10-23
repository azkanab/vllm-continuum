"""Generate synthetic agent trace jobs (contract C3) with a mix of fast (<=2 s) and slow tools.

  python dagshare_exp/traces/make_synthetic.py --jobs 200 --seed 0 --out dagshare_exp/traces/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

TOOLS = {
    "ls": (0.2, 0.15), "cat": (0.2, 0.15), "grep": (0.5, 0.15), "sed": (0.3, 0.1),
    "find": (1.0, 0.1), "git": (0.8, 0.1), "python": (6.0, 0.05), "pytest": (15.0, 0.05),
    "pip": (25.0, 0.02),
}


def lognormal(rng: random.Random, mean: float, sigma: float) -> float:
    """Lognormal sample with the given mean."""
    return rng.lognormvariate(math.log(mean) - sigma * sigma / 2, sigma)


def make_job(rng: random.Random, job_id: str, max_context: int, min_turns: int, max_turns: int) -> dict:
    names = list(TOOLS)
    weights = [TOOLS[n][1] for n in names]
    n_turns = rng.randint(min_turns, max_turns)
    prompt = int(min(6000, max(800, lognormal(rng, 2500, 0.4))))
    turns = []
    for i in range(n_turns):
        out = int(min(1024, max(16, lognormal(rng, 180, 0.6))))
        last = i == n_turns - 1
        tool = None if last else rng.choices(names, weights)[0]
        tool_time = 0.0 if last else round(lognormal(rng, TOOLS[tool][0], 0.5), 3)
        tool_out = 0 if last else int(min(4000, max(8, lognormal(rng, 400, 0.9))))
        if not last and prompt + out + tool_out + 1024 > max_context:
            tool, tool_time, tool_out, last = None, 0.0, 0, True
        turns.append({"prompt_tokens": prompt, "output_tokens": out, "tool_name": tool,
                      "tool_time_s": tool_time, "tool_output_tokens": tool_out})
        if last:
            break
        prompt += out + tool_out
    return {"job_id": job_id, "source": "synthetic", "turns": turns}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-turns", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-context", type=int, default=24000, help="stop a job before its prompt exceeds this")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "synthetic.jsonl"))
    a = p.parse_args()
    rng = random.Random(a.seed)
    jobs = [make_job(rng, f"syn{i:04d}", a.max_context, a.min_turns, a.max_turns) for i in range(a.jobs)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    n_turns = sum(len(j["turns"]) for j in jobs)
    fast = sum(1 for j in jobs for t in j["turns"] if t["tool_name"] and TOOLS[t["tool_name"]][0] <= 2.0)
    print(f"wrote {len(jobs)} jobs, {n_turns} turns ({fast} fast-tool turns) to {a.out}")


if __name__ == "__main__":
    main()
