"""Tiny OpenAI-compatible /v1/completions server for testing loadgen without a GPU.

Streams exactly max_tokens tokens with a fixed per-token delay, echoes the DAGShare
request fields (C2) back under "dagshare", and optionally logs every request as JSONL.

  python -m dagshare_exp.mock_server --port 8000 --token-delay-s 0.01 --log-file /tmp/mock.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web

C2_FIELDS = ("tenant_id", "job_id", "is_last_step", "this_func_call", "last_func_call")


def make_app(token_delay_s: float = 0.005, prefill_s_per_1k: float = 0.01, model: str = "mock-model",
             log_file: str | None = None, max_concurrency: int = 0) -> web.Application:
    log = open(log_file, "a") if log_file else None
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
    counter = {"n": 0}

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="")

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            prompt_tokens = len(prompt)
        else:
            prompt_tokens = max(1, len(str(prompt)) // 4)
        max_tokens = int(body.get("max_tokens") or 16)
        echo = {k: body.get(k) for k in C2_FIELDS}
        counter["n"] += 1
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        if log:
            log.write(json.dumps({"t": time.time(), "id": req_id, "prompt_tokens": prompt_tokens,
                                  "max_tokens": max_tokens, "ignore_eos": body.get("ignore_eos"),
                                  "model": body.get("model"), **echo}) + "\n")
            log.flush()
        if sem:
            await sem.acquire()
        try:
            await asyncio.sleep(prefill_s_per_1k * prompt_tokens / 1000)
            base = {"id": req_id, "object": "text_completion", "created": int(time.time()),
                    "model": model, "dagshare": echo}
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                     "total_tokens": prompt_tokens + max_tokens}
            token_ids = [(prompt_tokens + i) % 30000 + 1000 for i in range(max_tokens)]
            if not body.get("stream"):
                await asyncio.sleep(token_delay_s * max_tokens)
                choice = {"index": 0, "text": " tok" * max_tokens, "finish_reason": "length",
                          "token_ids": token_ids if body.get("return_token_ids") else None}
                return web.json_response({**base, "choices": [choice], "usage": usage})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for i, tid in enumerate(token_ids):
                if i:
                    await asyncio.sleep(token_delay_s)
                choice = {"index": 0, "text": " tok", "finish_reason": "length" if i == max_tokens - 1 else None}
                if body.get("return_token_ids"):
                    choice["token_ids"] = [tid]
                await resp.write(f"data: {json.dumps({**base, 'choices': [choice]})}\n\n".encode())
            if (body.get("stream_options") or {}).get("include_usage"):
                await resp.write(f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            if sem:
                sem.release()

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token-delay-s", type=float, default=0.005)
    p.add_argument("--prefill-s-per-1k", type=float, default=0.01, help="prefill delay per 1000 prompt tokens")
    p.add_argument("--model", default="mock-model")
    p.add_argument("--log-file", default=None)
    p.add_argument("--max-concurrency", type=int, default=0, help="0 = unlimited")
    a = p.parse_args()
    web.run_app(make_app(a.token_delay_s, a.prefill_s_per_1k, a.model, a.log_file, a.max_concurrency),
                host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
