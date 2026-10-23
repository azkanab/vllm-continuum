# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def make(ds, **cfg):
    cfg.setdefault("alpha", 0.0)
    cfg.setdefault("beta", 0.0)
    return ds.TenantLedger(ds.DagShareConfig(**cfg), 4, 100, 10**8)


def cand(ds, rid, tenant, arrival, predicted=1):
    return ds.CandidateView(rid, tenant, None, arrival, predicted, predicted)


def charge(ds, ledger, slots_by_tenant):
    """Run each tenant on the given number of slots for 1 s; S = slots / 4."""
    running = [
        ds.ReqView(f"{t}{i}", t, None, 0.0, 0)
        for t, n in slots_by_tenant.items() for i in range(n)
    ]
    ledger.update(0.0, running, [])
    ledger.update(1.0, running, [])


def test_empty_is_blocked(ds):
    assert make(ds).pick([], 100, 0.0) == ds.PickResult(None, None, "blocked")


def test_lowest_score_wins(ds):
    ledger = make(ds)
    charge(ds, ledger, {"A": 2, "B": 1})
    res = ledger.pick([cand(ds, "a", "A", 0.0), cand(ds, "b", "B", 5.0)],
                      100, 2.0)
    assert res == ds.PickResult("b", "B", "lowest")


def test_oldest_within_tenant(ds):
    ledger = make(ds)
    res = ledger.pick([cand(ds, "late", "A", 3.0),
                       cand(ds, "early", "A", 1.0)], 100, 0.0)
    assert res.request_id == "early"


def test_tie_breaks_on_arrival(ds):
    ledger = make(ds)
    res = ledger.pick([cand(ds, "a", "A", 2.0), cand(ds, "b", "B", 1.0)],
                      100, 0.0)
    assert res.tenant_id == "B"


def test_round_robin_on_exact_ties(ds):
    ledger = make(ds)
    cands = [cand(ds, "a", "A", 1.0), cand(ds, "b", "B", 2.0)]
    picked = []
    for now in (10.0, 11.0, 12.0, 13.0):
        res = ledger.pick(cands, 100, now)
        picked.append(res.tenant_id)
        ledger.mark_released(res.tenant_id, now)
    assert picked == ["A", "B", "A", "B"]


def test_score_beats_last_picked(ds):
    ledger = make(ds)
    charge(ds, ledger, {"A": 1, "B": 2})
    ledger.mark_released("A", 5.0)
    res = ledger.pick([cand(ds, "a", "A", 1.0), cand(ds, "b", "B", 0.0)],
                      100, 6.0)
    assert res.tenant_id == "A"


def test_pick_does_not_mutate(ds):
    ledger = make(ds)
    cands = [cand(ds, "a", "A", 1.0), cand(ds, "b", "B", 2.0)]
    first = ledger.pick(cands, 100, 1.0)
    assert ledger.pick(cands, 100, 2.0) == first
    assert ledger.snapshot() == {}


def test_slack_within_delta(ds):
    ledger = make(ds, delta=0.3)
    charge(ds, ledger, {"A": 2, "B": 1})
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=5),
                       cand(ds, "b", "B", 0.0, predicted=50)], 10, 2.0)
    assert res == ds.PickResult("a", "A", "slack")


def test_slack_outside_delta_blocks(ds):
    ledger = make(ds, delta=0.2)
    charge(ds, ledger, {"A": 2, "B": 1})
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=5),
                       cand(ds, "b", "B", 0.0, predicted=50)], 10, 2.0)
    assert res == ds.PickResult(None, None, "blocked")


def test_strict_delta_allows_exact_tie(ds):
    ledger = make(ds)
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=50),
                       cand(ds, "b", "B", 1.0, predicted=5)], 10, 0.0)
    assert res == ds.PickResult("b", "B", "slack")


def test_slack_considers_only_oldest_request(ds):
    ledger = make(ds, delta=1.0)
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=50),
                       cand(ds, "b_old", "B", 1.0, predicted=50),
                       cand(ds, "b_new", "B", 2.0, predicted=1)], 10, 0.0)
    assert res == ds.PickResult(None, None, "blocked")


def test_slack_takes_first_fitting_in_order(ds):
    ledger = make(ds, delta=1.0)
    charge(ds, ledger, {"A": 1, "B": 2, "C": 3})
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=50),
                       cand(ds, "c", "C", 0.0, predicted=1),
                       cand(ds, "b", "B", 0.0, predicted=1)], 10, 2.0)
    assert res == ds.PickResult("b", "B", "slack")


@pytest.mark.parametrize("free,fits", [(10, True), (9, False)])
def test_fit_boundary(ds, free, fits):
    ledger = make(ds)
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=10)], free, 0.0)
    assert res.reason == ("lowest" if fits else "blocked")


def test_fit_check_off_ignores_memory(ds):
    ledger = make(ds, fit_check=False)
    charge(ds, ledger, {"A": 2, "B": 1})
    res = ledger.pick([cand(ds, "a", "A", 0.0, predicted=1),
                       cand(ds, "b", "B", 0.0, predicted=10**6)], 0, 2.0)
    assert res == ds.PickResult("b", "B", "lowest")
