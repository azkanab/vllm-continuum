# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

SLOTS = 4
BLOCKS = 100
BYTES_PER_BLOCK = 10**8  # 0.1 GB per block, so T_M = 10 GB


def make(ds, **cfg):
    return ds.TenantLedger(ds.DagShareConfig(**cfg), SLOTS, BLOCKS,
                           BYTES_PER_BLOCK)


def rv(ds, rid, tenant, blocks, tokens=0):
    return ds.ReqView(rid, tenant, None, 0.0, blocks, tokens)


def test_normalizers(ds):
    ledger = make(ds)
    assert ledger.T_C == 4
    assert ledger.T_M == pytest.approx(10.0)


def test_first_update_charges_nothing(ds):
    ledger = make(ds)
    ledger.update(100.0, [rv(ds, "a", "A", 10)], [rv(ds, "p", "B", 10)])
    assert ledger.snapshot() == {
        "A": {"C": 0.0, "M_act": 0.0, "M_idle": 0.0, "S": 0.0},
        "B": {"C": 0.0, "M_act": 0.0, "M_idle": 0.0, "S": 0.0},
    }


def _run_two_seconds(ledger, ds):
    running = [rv(ds, "a1", "A", 10), rv(ds, "a2", "A", 20),
               rv(ds, "b1", "B", 5)]
    pinned = [rv(ds, "bp", "B", 30)]
    ledger.update(10.0, running, pinned)
    ledger.update(12.0, running, pinned)


def test_hand_computed(ds):
    ledger = make(ds)
    _run_two_seconds(ledger, ds)
    snap = ledger.snapshot()
    # A: C = 2 slots * 2 s = 4; M_act = (1 + 2) GB * 2 s = 6; S = 4/4 + 6/10
    assert snap["A"] == pytest.approx({
        "C": 4.0, "M_act": 6.0, "M_idle": 0.0, "S": 1.6})
    # B: C = 2; M_act = 0.5 GB * 2 s = 1; M_idle = 3 GB * 2 s = 6
    assert snap["B"] == pytest.approx({
        "C": 2.0, "M_act": 1.0, "M_idle": 6.0, "S": 0.5 + 0.1 + 0.6})
    assert ledger.score("A") == pytest.approx(1.6)


def test_alpha_beta_weights(ds):
    ledger = make(ds, alpha=2.0, beta=0.5)
    _run_two_seconds(ledger, ds)
    assert ledger.score("A") == pytest.approx(1.0 + 2.0 * 0.6)
    assert ledger.score("B") == pytest.approx(0.5 + 2.0 * 0.1 + 0.5 * 0.6)


def test_beta_zero_ignores_idle(ds):
    ledger = make(ds, beta=0.0)
    _run_two_seconds(ledger, ds)
    assert ledger.snapshot()["B"]["M_idle"] == pytest.approx(6.0)
    assert ledger.score("B") == pytest.approx(0.6)


def test_compute_off(ds):
    ledger = make(ds, compute=False)
    _run_two_seconds(ledger, ds)
    snap = ledger.snapshot()
    assert snap["A"]["C"] == 0.0 and snap["B"]["C"] == 0.0
    assert ledger.score("A") == pytest.approx(0.6)
    assert ledger.score("B") == pytest.approx(0.7)


def test_tokens_share(ds):
    ledger = make(ds, compute_share="tokens")
    running = [rv(ds, "a1", "A", 0, 20), rv(ds, "a2", "A", 0, 10),
               rv(ds, "b1", "B", 0, 10)]
    ledger.update(0.0, running, [])
    ledger.update(2.0, running, [])
    assert ledger.snapshot()["A"]["C"] == pytest.approx(1.5)
    assert ledger.snapshot()["B"]["C"] == pytest.approx(0.5)


def test_tokens_share_zero_total(ds):
    ledger = make(ds, compute_share="tokens")
    running = [rv(ds, "a1", "A", 10, 0)]
    ledger.update(0.0, running, [])
    ledger.update(2.0, running, [])
    assert ledger.snapshot()["A"]["C"] == 0.0
    assert ledger.snapshot()["A"]["M_act"] == pytest.approx(2.0)


def test_negative_dt_clamped(ds):
    ledger = make(ds)
    running = [rv(ds, "a1", "A", 10)]
    ledger.update(5.0, running, [])
    ledger.update(3.0, running, [])
    assert ledger.snapshot()["A"]["C"] == 0.0
    ledger.update(4.0, running, [])
    assert ledger.snapshot()["A"]["C"] == pytest.approx(1.0)


def test_dt_uses_previous_update_even_if_empty(ds):
    ledger = make(ds)
    ledger.update(0.0, [], [])
    ledger.update(1.0, [], [])
    ledger.update(1.5, [rv(ds, "a1", "A", 0)], [])
    assert ledger.snapshot()["A"]["C"] == pytest.approx(0.5)


def test_unknown_tenant_scores_zero(ds):
    ledger = make(ds)
    assert ledger.score("nobody") == 0.0
    assert ledger.snapshot() == {}


def test_preemption_order(ds):
    ledger = make(ds, alpha=0.0, beta=0.0)
    running = [rv(ds, "a1", "A", 0), rv(ds, "a2", "A", 0),
               rv(ds, "b1", "B", 0), rv(ds, "c1", "C", 0)]
    ledger.update(0.0, running, [])
    ledger.update(1.0, running, [])
    ledger.mark_released("B", 5.0)
    ledger.mark_released("C", 7.0)
    assert ledger.preemption_order() == ["A", "C", "B"]
    ledger.mark_released("B", 9.0)
    assert ledger.preemption_order() == ["A", "B", "C"]
