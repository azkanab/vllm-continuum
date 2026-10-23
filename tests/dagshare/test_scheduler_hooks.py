# SPDX-License-Identifier: Apache-2.0
"""Scheduler hook tests for the dagshare policy (stream B)."""

import pytest

torch = pytest.importorskip("torch")

from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig,  # noqa: E402
                         VllmConfig)
from vllm.v1.core.estimate_with_func import ToolCallEstimator  # noqa: E402
from vllm.v1.core.sched import scheduler as sched_mod  # noqa: E402
from vllm.v1.core.sched.dagshare import PickResult  # noqa: E402
from vllm.v1.core.sched.request_queue import (  # noqa: E402
    DagShareRequestQueue, FCFSRequestQueue, SchedulingPolicy,
    create_request_queue)
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import (FullAttentionSpec,  # noqa: E402
                                        KVCacheConfig, KVCacheGroupSpec)
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from tests.v1.core.utils import create_requests  # noqa: E402

BLOCK_SIZE = 16


class FakeLedger:

    def __init__(self, cfg, total_slots, total_kv_blocks, bytes_per_block):
        self.init = (total_slots, total_kv_blocks, bytes_per_block)
        self.updates = []
        self.released = []
        self.pick_fn = None
        self.order = []

    def update(self, now, running, pinned):
        self.updates.append((list(running), list(pinned)))

    def pick(self, candidates, free_blocks, now):
        if self.pick_fn is not None:
            return self.pick_fn(candidates, free_blocks)
        c = min(candidates, key=lambda c: c.arrival_time)
        return PickResult(c.request_id, c.tenant_id, "lowest")

    def score(self, tenant):
        return 0.0

    def mark_released(self, tenant, now):
        self.released.append(tenant)

    def preemption_order(self):
        return list(self.order)

    def snapshot(self):
        return {}


class FakeLogger:

    def __init__(self, cfg):
        self.config = None
        self.events = []
        self.ledger_stats = []

    def write_config(self, info):
        self.config = info

    def maybe_log_ledger(self, now, ledger, stats):
        self.ledger_stats.append(stats)

    def event(self, **fields):
        self.events.append(fields)

    def names(self):
        return [e["event"] for e in self.events]


def make_scheduler(policy="dagshare", num_blocks=10000, max_num_seqs=16):
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs,
                                       max_num_batched_tokens=8192,
                                       max_model_len=8192,
                                       enable_chunked_prefill=True,
                                       policy=policy)
    model_config = ModelConfig(model="facebook/opt-125m",
                               trust_remote_code=True,
                               dtype="float16",
                               seed=42,
                               skip_tokenizer_init=True)
    cache_config = CacheConfig(block_size=BLOCK_SIZE,
                               gpu_memory_utilization=0.9,
                               swap_space=0,
                               cache_dtype="auto",
                               enable_prefix_caching=True)
    vllm_config = VllmConfig(scheduler_config=scheduler_config,
                             model_config=model_config,
                             cache_config=cache_config)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(BLOCK_SIZE, 1, 1, torch.float32,
                                               False))
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(vllm_config=vllm_config,
                     kv_cache_config=kv_cache_config,
                     log_stats=True,
                     structured_output_manager=StructuredOutputManager(
                         vllm_config))


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(sched_mod, "TenantLedger", FakeLedger)
    monkeypatch.setattr(sched_mod, "RunLogger", FakeLogger)
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("KV_RETENTION", raising=False)


def tagged_requests(tenants, num_tokens=32):
    reqs = create_requests(len(tenants),
                           num_tokens=num_tokens,
                           block_size=BLOCK_SIZE)
    for i, (req, tenant) in enumerate(zip(reqs, tenants)):
        req.tenant_id = tenant
        req.job_id = f"job{i}"
        req.arrival_time = float(i)
    return reqs


def test_dagshare_queue_chooser_and_fcfs_fallback():
    queue = create_request_queue(SchedulingPolicy.DAGSHARE)
    assert isinstance(queue, DagShareRequestQueue)
    reqs = tagged_requests(["a", "b", "c"])
    for r in reqs:
        queue.add_request(r)
    assert queue.peek_request() is reqs[0]
    queue.set_chooser(lambda waiting: waiting[2])
    assert queue.peek_request() is reqs[2]
    assert queue.pop_request() is reqs[2]
    queue.set_chooser(lambda waiting: None)
    assert queue.peek_request() is None
    assert [r.request_id for r in queue] == ["0", "1"]


def test_baseline_policies_have_no_dagshare_state(fakes):
    fcfs = make_scheduler("fcfs")
    assert isinstance(fcfs.waiting, FCFSRequestQueue)
    assert fcfs.dagshare_ledger is None and fcfs.retention_rule is None
    continuum = make_scheduler("continuum")
    assert continuum.dagshare_ledger is None
    assert continuum.retention_rule is None


def test_dagshare_init_builds_ledger_and_writes_config(fakes):
    s = make_scheduler(num_blocks=500, max_num_seqs=8)
    assert s.policy == SchedulingPolicy.DAGSHARE
    bytes_per_block = 2 * BLOCK_SIZE * 1 * 1 * 4
    assert s.dagshare_ledger.init == (8, 500, bytes_per_block)
    info = s.dagshare_logger.config
    assert info["policy"] == "dagshare"
    assert info["bytes_per_block"] == bytes_per_block
    assert info["num_gpu_blocks"] == 500
    assert s.retention_rule.kind == "ttl"


def test_chooser_overrides_fcfs_and_marks_release(fakes):
    s = make_scheduler(max_num_seqs=1)
    reqs = tagged_requests(["a", "b"])
    for r in reqs:
        s.add_request(r)
    s.dagshare_ledger.pick_fn = lambda cands, free: PickResult(
        "1", "b", "lowest")
    out = s.schedule()
    assert [r.req_id for r in out.scheduled_new_reqs] == ["1"]
    assert s.dagshare_ledger.released == ["b"]
    release = [e for e in s.dagshare_logger.events
               if e["event"] == "release"]
    assert release[0]["request_id"] == "1"
    assert release[0]["reason"] == "lowest"
    assert release[0]["gb"] > 0


def test_blocked_pick_releases_nothing_and_logs_once(fakes):
    s = make_scheduler()
    running, *queued = tagged_requests(["a", "b", "c"])
    s.add_request(running)
    s.schedule()
    for r in queued:
        s.add_request(r)
    s.dagshare_ledger.pick_fn = lambda cands, free: PickResult(
        None, None, "blocked")
    s.dagshare_logger.events.clear()
    out = s.schedule()
    assert not out.scheduled_new_reqs
    assert len(s.waiting) == 2
    assert s.dagshare_logger.names() == ["blocked"]
    s.schedule()
    assert s.dagshare_logger.names() == ["blocked", "blocked"]


def test_candidate_views_and_ledger_update(fakes):
    s = make_scheduler()
    seen = {}

    def pick(cands, free):
        seen["cands"] = list(cands)
        seen["free"] = free
        c = cands[0]
        return PickResult(c.request_id, c.tenant_id, "lowest")

    s.dagshare_ledger.pick_fn = pick
    reqs = tagged_requests(["a"], num_tokens=40)
    s.add_request(reqs[0])
    s.schedule()
    cand = seen["cands"][0]
    assert cand.blocks_needed_now == 3
    assert cand.blocks_predicted == 3 + s.dagshare_config.fit_margin_blocks
    assert seen["free"] > 0
    s.schedule()
    running, pinned = s.dagshare_ledger.updates[-1]
    assert running[0].tenant_id == "a" and running[0].blocks_held == 3
    assert running[0].scheduled_tokens == 40
    assert pinned == []
    assert s.dagshare_logger.ledger_stats[-1]["n_running"] == 1


def test_pin_on_finish_and_unpin_on_next_request(fakes):
    s = make_scheduler()
    first, second = tagged_requests(["a", "a"])
    second.job_id = first.job_id = "jobA"
    first.this_func_call = "ls"
    first.is_last_step = False
    s.add_request(first)
    s.schedule()
    s.finish_requests(first.request_id, RequestStatus.FINISHED_STOPPED)
    assert [r for r, _ in s.pinned_requests] == [first]
    assert "pin" in s.dagshare_logger.names()
    s.schedule()
    _, pinned = s.dagshare_ledger.updates[-1]
    assert pinned[0].blocks_held == 2
    second.is_last_step = True
    s.add_request(second)
    s.schedule()
    s.finish_requests(second.request_id, RequestStatus.FINISHED_STOPPED)
    unpins = [e for e in s.dagshare_logger.events if e["event"] == "unpin"]
    assert unpins[0]["reason"] == "next_request_done"
    assert not s.pinned_requests


def test_unpin_and_victim_follow_preemption_order(fakes):
    s = make_scheduler()
    reqs = tagged_requests(["a", "b", "b", "c"])
    for r in reqs:
        s.add_request(r)
    s.schedule()
    s.dagshare_ledger.order = ["b", "a", "c"]
    victim = s._dagshare_pick_victim(reqs[0])
    assert victim is reqs[2]
    assert s._dagshare_pick_victim(reqs[0]) is not reqs[0]
    s.running.remove(reqs[0])
    s.running.remove(reqs[3])
    s.pin_request(reqs[0], 5.0)
    s.pin_request(reqs[3], 5.0)
    s.dagshare_ledger.order = ["c", "a"]
    assert s._dagshare_unpin_one("preempt") is reqs[3]
    assert [r for r, _ in s.pinned_requests] == [reqs[0]]


def test_running_preemption_unpins_before_preempting(fakes):
    s = make_scheduler(num_blocks=6)
    a, b = tagged_requests(["a", "b"], num_tokens=32)
    s.add_request(a)
    s.schedule()
    s.running.remove(a)
    s.pin_request(a, 100.0)
    s.add_request(b)
    s.schedule()
    assert b in s.running
    s.dagshare_ledger.order = ["a", "b"]
    b.append_output_token_ids([1] * 33)
    b.num_computed_tokens = b.num_tokens - 1
    s.schedule()
    names = s.dagshare_logger.names()
    assert "unpin" in names
    assert not s.pinned_requests


def test_estimator_keeps_client_this_func_call():
    est = ToolCallEstimator()
    req = tagged_requests(["a"])[0]
    req.this_func_call = "pytest"
    est.request_arrives(req)
    est.request_finished(req)
    assert req.this_func_call == "pytest"
    assert est.job_to_history[req.job_id][-1]["func_call"] == "pytest"


def test_blocked_with_nothing_running_unpins_then_falls_back(fakes):
    s = make_scheduler()
    a, b = tagged_requests(["a", "b"])
    s.add_request(a)
    s.schedule()
    s.running.remove(a)
    s.pin_request(a, 100.0)
    s.add_request(b)
    s.dagshare_ledger.pick_fn = lambda cands, free: PickResult(
        None, None, "blocked")
    out = s.schedule()
    assert [r.req_id for r in out.scheduled_new_reqs] == [b.request_id]
    assert not s.pinned_requests
    unpin = [e for e in s.dagshare_logger.events if e["event"] == "unpin"]
    assert unpin[0]["reason"] == "blocked_idle"
    release = [e for e in s.dagshare_logger.events
               if e["event"] == "release"]
    assert release[-1]["reason"] == "idle_fallback"
