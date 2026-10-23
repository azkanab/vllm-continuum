"""Send a few short completions so CUDA graphs and caches are warm before measuring."""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one(base_url: str, model: str, i: int) -> bool:
    body = {
        "model": model,
        "prompt": f"Warmup request {i}: write one sentence about GPUs.",
        "max_tokens": 32,
        "temperature": 0.0,
        "tenant_id": "warmup",
        "job_id": f"warmup-{i}",
        "is_last_step": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"warmup request {i} failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        ok = sum(pool.map(lambda i: one(args.base_url, args.model, i), range(args.n)))
    print(f"warmup: {ok}/{args.n} requests succeeded", file=sys.stderr)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
