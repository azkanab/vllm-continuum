# SPDX-License-Identifier: Apache-2.0
"""DAGShare: per-tenant compute + memory ledger scheduling (contract C1 in docs/dagshare/IMPLEMENTATION.md)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Optional

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class DagShareConfig:
    compute: bool = True
    compute_share: Literal["slot", "tokens"] = "slot"
    alpha: float = 1.0
    beta: float = 1.0
    fit_check: bool = True
    fit_estimate: Literal["margin", "p90"] = "margin"
    fit_margin_blocks: int = 16
    delta: float = 0.0
    charge_mode: Literal["measured", "predicted",
                         "predicted_corrected"] = "measured"
    evict_overserved: bool = False
    log_interval_ms: int = 500
    run_output_dir: str = "./results/run"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "DagShareConfig":
        raise NotImplementedError


@dataclass(frozen=True)
class RetentionRule:
    kind: Literal["none", "ttl", "until_done"]
    seconds: float = 0.0
    fast_tools_only: bool = True

    @classmethod
    def from_env(cls, policy: str,
                 env: Mapping[str, str] = os.environ) -> "RetentionRule":
        raise NotImplementedError

    def pin_seconds(self, *, this_func_call: Optional[str],
                    tool_avg_s: Optional[float],
                    is_last_step: Optional[bool]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ReqView:
    request_id: str
    tenant_id: str
    job_id: Optional[str]
    arrival_time: float
    blocks_held: int
    scheduled_tokens: int = 0


@dataclass(frozen=True)
class CandidateView:
    request_id: str
    tenant_id: str
    job_id: Optional[str]
    arrival_time: float
    blocks_needed_now: int
    blocks_predicted: int


@dataclass(frozen=True)
class PickResult:
    request_id: Optional[str]
    tenant_id: Optional[str]
    reason: Literal["lowest", "slack", "blocked"]


class TenantLedger:

    def __init__(self, cfg: DagShareConfig, total_slots: int,
                 total_kv_blocks: int, bytes_per_block: int) -> None:
        raise NotImplementedError

    def update(self, now: float, running: Sequence[ReqView],
               pinned: Sequence[ReqView]) -> None:
        raise NotImplementedError

    def score(self, tenant: str) -> float:
        raise NotImplementedError

    def pick(self, candidates: Sequence[CandidateView], free_blocks: int,
             now: float) -> PickResult:
        raise NotImplementedError

    def mark_released(self, tenant: str, now: float) -> None:
        raise NotImplementedError

    def preemption_order(self) -> list[str]:
        raise NotImplementedError

    def snapshot(self) -> dict:
        raise NotImplementedError


class RunLogger:

    def __init__(self, cfg: DagShareConfig) -> None:
        raise NotImplementedError

    def write_config(self, info: dict) -> None:
        raise NotImplementedError

    def maybe_log_ledger(self, now: float, ledger: TenantLedger,
                         stats: dict) -> None:
        raise NotImplementedError

    def event(self, **fields) -> None:
        raise NotImplementedError
