"""End-to-end loadgen test against mock_server (no GPU)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "dagshare_exp"
C4_FIELDS = {"tenant", "job_id", "turn", "send_t", "first_token_t", "finish_t",
             "prompt_tokens", "output_tokens", "status", "warmup"}
C2_FIELDS = {"tenant_id", "job_id", "is_last_step", "this_func_call", "last_func_call"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_server(tmp_path_factory):
    log = tmp_path_factory.mktemp("mock") / "server.jsonl"
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "dagshare_exp.mock_server", "--port", str(port),
                             "--token-delay-s", "0.002", "--log-file", str(log)], cwd=ROOT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("mock server did not start")
    yield url, log
    proc.terminate()
    proc.wait(timeout=10)


def write_inputs(tmp: Path) -> Path:
    traces = tmp / "traces.jsonl"
    with open(traces, "w") as f:
        for j in range(6):
            turns = [{"prompt_tokens": 200 + 40 * j, "output_tokens": 12, "tool_name": "ls",
                      "tool_time_s": 0.2, "tool_output_tokens": 30},
                     {"prompt_tokens": 0, "output_tokens": 8, "tool_name": "pytest",
                      "tool_time_s": 0.4, "tool_output_tokens": 50},
                     {"prompt_tokens": 0, "output_tokens": 6, "tool_name": None,
                      "tool_time_s": 0.0, "tool_output_tokens": 0}]
            f.write(json.dumps({"job_id": f"t{j}", "source": "synthetic", "turns": turns}) + "\n")
    chat = tmp / "chat.jsonl"
    with open(chat, "w") as f:
        for i in range(20):
            f.write(json.dumps({"prompt": f"question {i} " * 10, "output_tokens": 5 + i}) + "\n")
    smoke = yaml.safe_load((EXP / "configs/workloads/smoke.yaml").read_text())
    for t in smoke["tenants"]:
        if t["kind"] == "agent":
            t.update(traces=str(traces), jobs_in_flight=3, tool_time_scale=0.5)
        else:
            t.update(source=str(chat), rate_rps=4.0)
    wl = tmp / "workload.yaml"
    wl.write_text(yaml.safe_dump(smoke))
    return wl


def test_loadgen_against_mock(mock_server, tmp_path):
    url, server_log = mock_server
    wl = write_inputs(tmp_path)
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, "-m", "dagshare_exp.loadgen", "--workload", str(wl),
                          "--base-url", url, "--model", "mock-model", "--seed", "1", "--out", str(out),
                          "--duration-s", "8", "--warmup-s", "2", "--drain-s", "5"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "LOADGEN_SUMMARY" in res.stdout

    rows = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        assert C4_FIELDS <= r.keys()
        if r["status"] == "ok":
            assert r["send_t"] <= r["first_token_t"] <= r["finish_t"]
            assert r["output_tokens"] > 0 and r["prompt_tokens"] > 0
    assert {r["tenant"] for r in rows} == {"agent", "chat"}
    assert any(r["warmup"] for r in rows) and any(not r["warmup"] for r in rows)

    jobs = defaultdict(list)
    for r in rows:
        if r["tenant"] == "agent":
            jobs[r["job_id"]].append(r)
    assert any(len(v) == 3 for v in jobs.values()), "no agent job completed"
    for v in jobs.values():
        v.sort(key=lambda r: r["turn"])
        assert [r["turn"] for r in v] == list(range(len(v)))
        for a, b in zip(v, v[1:]):
            assert a["finish_t"] <= b["send_t"], "agent turns overlap"
            assert b["prompt_tokens"] >= a["prompt_tokens"] + a["output_tokens"]
    chat_ids = [r["job_id"] for r in rows if r["tenant"] == "chat"]
    assert len(chat_ids) == len(set(chat_ids))

    reqs = [json.loads(line) for line in server_log.read_text().splitlines()]
    assert all(C2_FIELDS <= r.keys() for r in reqs)
    assert all(r["ignore_eos"] is True for r in reqs)
    by_job = defaultdict(list)
    for r in reqs:
        by_job[r["job_id"]].append(r)
        assert r["tenant_id"] in ("agent", "chat")
    for job_id, v in by_job.items():
        if job_id.startswith("chat-"):
            assert len(v) == 1 and v[0]["is_last_step"] is True and v[0]["this_func_call"] is None
            continue
        assert [r["this_func_call"] for r in v] == ["ls", "pytest", None][: len(v)]
        assert [r["last_func_call"] for r in v] == [None, "ls", "pytest"][: len(v)]
        assert [r["is_last_step"] for r in v] == [False, False, True][: len(v)]


def test_load_workload_resolves_repo_configs():
    sys.path.insert(0, str(ROOT))
    from dagshare_exp.loadgen import load_workload

    for name in ("smoke", "main"):
        wl = load_workload(EXP / f"configs/workloads/{name}.yaml")
        kinds = {t.kind for t in wl.tenants}
        assert kinds == {"agent", "chat"}
        agent = next(t for t in wl.tenants if t.kind == "agent")
        assert agent.cfg["jobs_in_flight"] > 0 and Path(agent.cfg["traces"]).is_absolute()


def test_synthetic_traces_match_c3():
    path = EXP / "traces/synthetic.jsonl"
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    assert jobs
    for j in jobs:
        assert j["source"] == "synthetic" and j["turns"]
        assert j["turns"][-1]["tool_name"] is None
        for a, b in zip(j["turns"], j["turns"][1:]):
            assert b["prompt_tokens"] == a["prompt_tokens"] + a["output_tokens"] + a["tool_output_tokens"]
            assert a["tool_name"] and a["tool_time_s"] > 0
