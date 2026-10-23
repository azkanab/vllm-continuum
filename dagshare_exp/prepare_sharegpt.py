"""Download ShareGPT and convert it to chat-tenant JSONL: {"prompt": str, "output_tokens": int, "prompt_tokens": int}.

Each conversation contributes its first human turn as the prompt and the following gpt turn's
length as output_tokens. Token counts use the HF tokenizer given by --tokenizer (needs
`transformers`); without it they use the heuristic ceil(chars / 4).

  python dagshare_exp/prepare_sharegpt.py --out dagshare_exp/data/sharegpt.jsonl \
      [--input ShareGPT_V3_unfiltered_cleaned_split.json] [--tokenizer meta-llama/Llama-3.1-8B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{FILE}"
EXP_DIR = Path(__file__).resolve().parent


def download(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=REPO, filename=FILE, repo_type="dataset"))
    except ImportError:
        pass
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / FILE
    if not dest.exists():
        print(f"downloading {URL} -> {dest}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(dest)
    return dest


def make_counter(tokenizer_name: str | None):
    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda text: len(tok(text, add_special_tokens=False).input_ids), f"tokenizer:{tokenizer_name}"
    return lambda text: max(1, math.ceil(len(text) / 4)), "chars/4"


def convert(conversations: list[dict], count, min_tokens: int, max_prompt: int, max_total: int,
            limit: int = 0) -> list[dict]:
    rows = []
    for conv in conversations:
        if limit and len(rows) >= limit:
            break
        turns = conv.get("conversations") or []
        for i in range(len(turns) - 1):
            if turns[i].get("from") == "human" and turns[i + 1].get("from") == "gpt":
                prompt, answer = turns[i]["value"], turns[i + 1]["value"]
                pt, ot = count(prompt), count(answer)
                if pt >= min_tokens and ot >= min_tokens and pt <= max_prompt and pt + ot <= max_total:
                    rows.append({"prompt": prompt, "output_tokens": ot, "prompt_tokens": pt})
                break
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None, help="local ShareGPT JSON (skips the download)")
    p.add_argument("--out", default=str(EXP_DIR / "data" / "sharegpt.jsonl"))
    p.add_argument("--cache-dir", default=str(EXP_DIR / "data" / "raw"))
    p.add_argument("--tokenizer", default=None, help="HF tokenizer for exact counts (else chars/4)")
    p.add_argument("--limit", type=int, default=10000, help="keep at most this many rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-tokens", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-total-tokens", type=int, default=2048)
    a = p.parse_args()
    src = Path(a.input) if a.input else download(Path(a.cache_dir))
    data = json.loads(src.read_text())
    random.Random(a.seed).shuffle(data)
    count, method = make_counter(a.tokenizer)
    rows = convert(data, count, a.min_tokens, a.max_prompt_tokens, a.max_total_tokens, a.limit)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} prompts to {a.out} (token counts: {method})")


if __name__ == "__main__":
    main()
