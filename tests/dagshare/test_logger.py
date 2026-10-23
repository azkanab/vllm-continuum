# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def make(ds, out_dir, **cfg):
    cfg = ds.DagShareConfig(run_output_dir=str(out_dir), **cfg)
    ledger = ds.TenantLedger(cfg, 4, 100, 10**8)
    return ds.RunLogger(cfg), ledger


def test_creates_dir_and_writes_config(ds, tmp_path):
    out = tmp_path / "a" / "b"
    log, _ = make(ds, out, beta=0.0)
    log.write_config({"policy": "dagshare", "env": {"DAGSHARE_BETA": "0"}})
    data = json.loads((out / "config.json").read_text())
    assert data["policy"] == "dagshare"
    assert data["env"] == {"DAGSHARE_BETA": "0"}
    assert data["dagshare"]["beta"] == 0.0


def test_ledger_interval(ds, tmp_path):
    log, ledger = make(ds, tmp_path, log_interval_ms=500)
    ledger.update(0.0, [ds.ReqView("r", "A", "j", 0.0, 10)], [])
    for now in (0.0, 0.2, 0.5, 0.9, 1.0):
        log.maybe_log_ledger(now, ledger, {"n_running": 1, "n_waiting": 0})
    rows = read_jsonl(tmp_path / "ledger.jsonl")
    assert [r["t"] for r in rows] == [0.0, 0.5, 1.0]
    assert rows[0] == {
        "t": 0.0,
        "tenants": {"A": {"C": 0.0, "M_act": 0.0, "M_idle": 0.0, "S": 0.0}},
        "n_running": 1,
        "n_waiting": 0,
    }


def test_event_adds_t(ds, tmp_path):
    log, _ = make(ds, tmp_path)
    log.event(event="pin", request_id="r1", gb=0.5)
    log.event(event="unpin", request_id="r1", t=42.0)
    rows = read_jsonl(tmp_path / "events.jsonl")
    assert isinstance(rows[0]["t"], float)
    assert rows[0]["event"] == "pin" and rows[0]["gb"] == 0.5
    assert rows[1]["t"] == 42.0


def test_never_raises_on_unwritable_dir(ds, tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    with caplog.at_level(logging.WARNING):
        log, ledger = make(ds, blocker / "sub")
        log.write_config({"x": object()})
        log.maybe_log_ledger(0.0, ledger, {})
        log.event(event="pin")
    assert "cannot write" in caplog.text
    assert blocker.read_text() == "not a dir"
