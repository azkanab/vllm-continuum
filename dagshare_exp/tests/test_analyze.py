import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dagshare_exp import analyze  # noqa: E402
from dagshare_exp.fixtures import make_fixtures  # noqa: E402


@pytest.fixture(scope="module")
def fixtures_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("results")
    make_fixtures.main(root)
    return root


@pytest.fixture(scope="module")
def outputs(fixtures_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    res = analyze.run(fixtures_root, out)
    return out, res


def _write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_parse_run_name():
    assert analyze.parse_run_name("A3_until_done60_seed2") == {
        "run": "A3_until_done60_seed2", "preset": "A3", "retention": "until_done60",
        "retention_kind": "until_done", "seed": 2}
    assert analyze.parse_run_name("vllm_none_seed1")["retention_kind"] == "none"
    assert analyze.parse_run_name("continuum_ttl2_seed3")["preset"] == "continuum"
    assert analyze.parse_run_name("not_a_run") is None


def test_outputs_exist_with_columns(outputs):
    out, res = outputs
    for name in ("summary.csv", "aggregate.csv", "comparisons.csv"):
        assert (out / name).exists()
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 20
    expected = {
        "preset", "retention", "seed", "ttft_p50_chat", "ttft_p99_chat", "e2e_p50_chat",
        "e2e_p99_chat", "ttft_p99_agent", "job_ct_p50", "job_ct_p99", "job_ct_mean",
        "throughput_req_s", "throughput_tok_s", "compute_share_agent", "memtime_share_chat",
        "idle_gbs_agent", "jain_service", "n_pin", "n_unpin", "n_preempt",
        "prefill_recomputes", "n_unfinished"}
    assert expected <= set(summary.columns)
    agg = pd.read_csv(out / "aggregate.csv")
    assert len(agg) == 10
    assert (agg["n_seeds"] == 2).all()
    assert {"ttft_p99_chat_mean", "ttft_p99_chat_std", "jain_service_mean"} <= set(agg.columns)
    assert len(res["plots"]) == 5
    assert all(p.stat().st_size > 0 for p in res["plots"])


def test_no_ledger_runs_have_nan_shares(outputs):
    summary = outputs[1]["summary"]
    vllm = summary[summary["preset"] == "vllm"]
    assert vllm["jain_service"].isna().all()
    assert vllm["ttft_p99_chat"].notna().all()
    a3 = summary[summary["preset"] == "A3"]
    assert a3["jain_service"].between(0.5, 1.0).all()
    shares = a3["compute_share_agent"] + a3["compute_share_chat"]
    assert shares.round(9).eq(1.0).all()


def test_comparison_rows(outputs):
    comp = pd.read_csv(outputs[0] / "comparisons.csv")
    pairs = set(zip(comp["comparison"], comp["retention_kind"]))
    assert pairs == {("A2_vs_A3", "ttl"), ("A2_vs_A3", "until_done"),
                     ("continuum_vs_A3", "ttl"), ("A0_vs_A3", "until_done")}
    row = comp[(comp["comparison"] == "A2_vs_A3") & (comp["retention_kind"] == "ttl")
               & (comp["metric"] == "ttft_p99_chat")].iloc[0]
    assert row["diff_b_minus_a"] == pytest.approx(row["mean_b"] - row["mean_a"])
    assert row["ci95_low"] <= row["diff_b_minus_a"] <= row["ci95_high"]
    assert row["n_a"] == row["n_b"] == 2


def test_hand_computed_p99_and_jain(tmp_path):
    run = tmp_path / "A3_ttl2_seed1"
    run.mkdir()
    reqs = [{"tenant": "chat", "job_id": f"c{k}", "turn": 0, "send_t": 1000.0,
             "first_token_t": 1000.0 + 0.1 * k, "finish_t": 1001.0 + 0.1 * k,
             "prompt_tokens": 10, "output_tokens": 5, "status": "ok"} for k in range(1, 11)]
    reqs.append({"tenant": "chat", "job_id": "w", "turn": 0, "send_t": 990.0,
                 "first_token_t": 999.0, "finish_t": 999.5, "prompt_tokens": 1,
                 "output_tokens": 1, "status": "ok", "warmup": True})
    reqs.append({"tenant": "chat", "job_id": "x", "turn": 0, "send_t": 1000.0,
                 "first_token_t": None, "finish_t": None, "prompt_tokens": 1,
                 "output_tokens": 0, "status": "error"})
    reqs += [{"tenant": "agent", "job_id": "j1", "turn": i, "send_t": 1000.0 + 3 * i,
              "first_token_t": 1000.5 + 3 * i, "finish_t": 1001.0 + 3 * i,
              "prompt_tokens": 100, "output_tokens": 10, "status": "ok"} for i in range(3)]
    _write(run / "requests.jsonl", reqs)
    _write(run / "ledger.jsonl", [
        {"t": 1000.0, "tenants": {"agent": {"C": 0, "M_act": 0, "M_idle": 0, "S": 0},
                                  "chat": {"C": 0, "M_act": 0, "M_idle": 0, "S": 0}},
         "gb_running": 0, "gb_pinned": 0, "n_running": 0, "n_waiting": 0, "n_pinned": 0},
        {"t": 1010.0, "tenants": {"agent": {"C": 3, "M_act": 1, "M_idle": 1, "S": 1},
                                  "chat": {"C": 1, "M_act": 2, "M_idle": 0, "S": 1}},
         "gb_running": 1, "gb_pinned": 1, "n_running": 1, "n_waiting": 0, "n_pinned": 1}])
    _write(run / "events.jsonl", [
        {"t": 1001.0, "event": "pin", "request_id": "r", "job_id": "j1", "tenant_id": "agent",
         "gb": 0.1, "reason": "tool"},
        {"t": 1002.0, "event": "unpin", "request_id": "r", "job_id": "j1", "tenant_id": "agent",
         "gb": 0.1, "reason": "tool"},
        {"t": 1003.0, "event": "preempt", "request_id": "q", "job_id": None,
         "tenant_id": "chat", "gb": 0.1, "reason": "oom"}])

    row = analyze.summarize(tmp_path).iloc[0]
    # numpy linear percentile over 0.1..1.0: index 0.99*9 = 8.91 -> 0.9 + 0.91*0.1
    assert row["ttft_p99_chat"] == pytest.approx(0.991)
    assert row["ttft_p50_chat"] == pytest.approx(0.55)
    # x = C share + mem share = (0.75+0.5, 0.25+0.5); J = 2^2 / (2*(1.25^2+0.75^2))
    assert row["jain_service"] == pytest.approx(4 / (2 * (1.25 ** 2 + 0.75 ** 2)))
    assert row["compute_share_agent"] == pytest.approx(0.75)
    assert row["memtime_share_agent"] == pytest.approx(0.5)
    assert row["idle_gbs_agent"] == pytest.approx(1.0)
    assert row["job_ct_mean"] == pytest.approx(7.0)
    assert row["n_unfinished"] == 1
    assert row["n_requests"] == 14
    assert (row["n_pin"], row["n_unpin"], row["n_preempt"], row["prefill_recomputes"]) == (1, 1, 1, 1)


def test_missing_ledger_still_produces_row(fixtures_root, tmp_path):
    src = fixtures_root / "A3_ttl2_seed1"
    dst = tmp_path / "A3_ttl2_seed1"
    shutil.copytree(src, dst)
    (dst / "ledger.jsonl").unlink()
    bad = tmp_path / "A2_ttl2_seed1"
    bad.mkdir()
    (bad / "requests.jsonl").write_text("{not json\n")
    summary = analyze.summarize(tmp_path)
    assert set(summary["run"]) == {"A3_ttl2_seed1", "A2_ttl2_seed1"}
    a3 = summary.set_index("run").loc["A3_ttl2_seed1"]
    assert pd.isna(a3["jain_service"])
    assert a3["ttft_p99_chat"] > 0
    assert a3["n_pin"] > 0


def test_cli(fixtures_root, tmp_path):
    analyze.main([str(fixtures_root), "--out", str(tmp_path / "o"), "--no-plots"])
    assert (tmp_path / "o" / "comparisons.csv").exists()
