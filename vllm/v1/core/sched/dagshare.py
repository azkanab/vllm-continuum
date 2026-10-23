# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DAGShare: per-tenant compute + memory ledger scheduling (contract C1 in docs/dagshare/IMPLEMENTATION.md)."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "default"

_ON = ("on", "true", "1", "yes")
_OFF = ("off", "false", "0", "no")
_RETENTION_POLICIES = ("continuum", "dagshare")


def _get(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _get(env, name)
    if value is None:
        return default
    if value.lower() in _ON:
        return True
    if value.lower() in _OFF:
        return False
    raise ValueError(f"{name} must be on/off, got {value!r}")


def _parse_choice(env: Mapping[str, str], name: str, choices: tuple[str, ...],
                  default: str) -> str:
    value = _get(env, name)
    if value is None:
        return default
    if value.lower() not in choices:
        raise ValueError(
            f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value.lower()


def _to_nonneg_float(name: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite float >= 0, got {value!r}")
    return number


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _get(env, name)
    return default if value is None else _to_nonneg_float(name, value)


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _get(env, name)
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


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
        """Parse the spec section 6 settings; raise ValueError on bad values."""
        d = cls()
        return cls(
            compute=_parse_bool(env, "DAGSHARE_COMPUTE", d.compute),
            compute_share=_parse_choice(env, "DAGSHARE_COMPUTE_SHARE",
                                        ("slot", "tokens"),
                                        d.compute_share),
            alpha=_parse_float(env, "DAGSHARE_ALPHA", d.alpha),
            beta=_parse_float(env, "DAGSHARE_BETA", d.beta),
            fit_check=_parse_bool(env, "DAGSHARE_FIT_CHECK", d.fit_check),
            fit_estimate=_parse_choice(env, "DAGSHARE_FIT_ESTIMATE",
                                       ("margin", "p90"), d.fit_estimate),
            fit_margin_blocks=_parse_int(env, "DAGSHARE_FIT_MARGIN",
                                         d.fit_margin_blocks),
            delta=_parse_float(env, "DAGSHARE_DELTA", d.delta),
            charge_mode=_parse_choice(
                env, "DAGSHARE_CHARGE_MODE",
                ("measured", "predicted", "predicted_corrected"),
                d.charge_mode),
            evict_overserved=_parse_bool(env, "DAGSHARE_EVICT_OVERSERVED",
                                         d.evict_overserved),
            log_interval_ms=_parse_int(env, "DAGSHARE_LOG_INTERVAL_MS",
                                       d.log_interval_ms),
            run_output_dir=_get(env, "RUN_OUTPUT_DIR") or d.run_output_dir,
        )


@dataclass(frozen=True)
class RetentionRule:
    kind: Literal["none", "ttl", "until_done"]
    seconds: float = 0.0
    fast_tools_only: bool = True

    @classmethod
    def from_env(cls, policy: str,
                 env: Mapping[str, str] = os.environ) -> "RetentionRule":
        """Parse KV_RETENTION and KV_RETENTION_FAST_TOOLS_ONLY for a policy."""
        fast_tools_only = _parse_bool(env, "KV_RETENTION_FAST_TOOLS_ONLY",
                                      True)
        value = _get(env, "KV_RETENTION")
        if value is None:
            if str(policy).lower() in _RETENTION_POLICIES:
                return cls("ttl", 2.0, fast_tools_only)
            return cls("none", 0.0, fast_tools_only)
        kind, sep, arg = value.partition(":")
        kind = kind.strip().lower()
        if kind == "none" and not sep:
            return cls("none", 0.0, fast_tools_only)
        if kind in ("ttl", "until_done") and sep and arg.strip():
            seconds = _to_nonneg_float("KV_RETENTION", arg.strip())
            return cls(kind, seconds, fast_tools_only)
        raise ValueError("KV_RETENTION must be none, ttl:<s> or "
                         f"until_done:<cap>, got {value!r}")

    def pin_seconds(self, *, this_func_call: Optional[str],
                    tool_avg_s: Optional[float],
                    is_last_step: Optional[bool]) -> float:
        """Seconds to keep a finished request's KV pinned (0 = free now)."""
        if is_last_step or self.kind == "none" or this_func_call is None:
            return 0.0
        if self.kind == "ttl":
            avg = tool_avg_s or 0.0
            if self.fast_tools_only and avg > self.seconds:
                return 0.0
        return self.seconds


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


@dataclass
class _TenantState:
    C: float = 0.0
    M_act: float = 0.0
    M_idle: float = 0.0
    last_picked: float = -math.inf


class TenantLedger:
    """Measured per-tenant compute and memory usage (spec section 4.3)."""

    def __init__(self, cfg: DagShareConfig, total_slots: int,
                 total_kv_blocks: int, bytes_per_block: int) -> None:
        self.cfg = cfg
        self.total_slots = total_slots
        self.total_kv_blocks = total_kv_blocks
        self.bytes_per_block = bytes_per_block
        self.T_C = float(total_slots)
        self.T_M = total_kv_blocks * bytes_per_block / 1e9
        self._tenants: dict[str, _TenantState] = {}
        self._last_update: Optional[float] = None

    def _state(self, tenant: str) -> _TenantState:
        state = self._tenants.get(tenant)
        if state is None:
            state = self._tenants[tenant] = _TenantState()
        return state

    def _gb(self, blocks: int) -> float:
        return blocks * self.bytes_per_block / 1e9

    def update(self, now: float, running: Sequence[ReqView],
               pinned: Sequence[ReqView]) -> None:
        dt = 0.0 if self._last_update is None else max(
            0.0, now - self._last_update)
        self._last_update = now
        total_tokens = sum(r.scheduled_tokens for r in running)
        for req in running:
            state = self._state(req.tenant_id)
            if self.cfg.compute:
                if self.cfg.compute_share == "tokens":
                    share = (req.scheduled_tokens /
                             total_tokens if total_tokens > 0 else 0.0)
                else:
                    share = 1.0
                state.C += share * dt
            state.M_act += self._gb(req.blocks_held) * dt
        for req in pinned:
            self._state(req.tenant_id).M_idle += self._gb(
                req.blocks_held) * dt

    def score(self, tenant: str) -> float:
        state = self._tenants.get(tenant)
        if state is None:
            return 0.0
        s = 0.0
        if self.cfg.compute and self.T_C > 0:
            s += state.C / self.T_C
        if self.T_M > 0:
            s += (self.cfg.alpha * state.M_act +
                  self.cfg.beta * state.M_idle) / self.T_M
        return s

    def _last_picked(self, tenant: str) -> float:
        state = self._tenants.get(tenant)
        return -math.inf if state is None else state.last_picked

    def pick(self, candidates: Sequence[CandidateView], free_blocks: int,
             now: float) -> PickResult:
        """Choose the next request to release; never mutates state."""
        oldest: dict[str, CandidateView] = {}
        for cand in candidates:
            best = oldest.get(cand.tenant_id)
            if best is None or cand.arrival_time < best.arrival_time:
                oldest[cand.tenant_id] = cand
        if not oldest:
            return PickResult(None, None, "blocked")
        scores = {tenant: self.score(tenant) for tenant in oldest}
        order = sorted(oldest,
                       key=lambda t: (scores[t], self._last_picked(t),
                                      oldest[t].arrival_time, t))
        lowest = order[0]
        if (not self.cfg.fit_check
                or oldest[lowest].blocks_predicted <= free_blocks):
            return PickResult(oldest[lowest].request_id, lowest, "lowest")
        s_max = scores[lowest] + self.cfg.delta
        for tenant in order[1:]:
            if scores[tenant] > s_max:
                break
            if oldest[tenant].blocks_predicted <= free_blocks:
                return PickResult(oldest[tenant].request_id, tenant, "slack")
        return PickResult(None, None, "blocked")

    def mark_released(self, tenant: str, now: float) -> None:
        self._state(tenant).last_picked = now

    def preemption_order(self) -> list[str]:
        """Tenants by score, highest first; ties: most recently picked first."""
        return sorted(self._tenants,
                      key=lambda t: (-self.score(t),
                                     -self._tenants[t].last_picked, t))

    def snapshot(self) -> dict:
        return {
            tenant: {
                "C": state.C,
                "M_act": state.M_act,
                "M_idle": state.M_idle,
                "S": self.score(tenant),
            }
            for tenant, state in self._tenants.items()
        }


class RunLogger:
    """Writes config.json, ledger.jsonl and events.jsonl; never raises."""

    def __init__(self, cfg: DagShareConfig) -> None:
        self.cfg = cfg
        self.out_dir = cfg.run_output_dir
        self._interval_s = cfg.log_interval_ms / 1000.0
        self._last_ledger_t: Optional[float] = None
        self._warned: set[str] = set()
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except OSError as e:
            self._warn(self.out_dir, e)

    def _warn(self, path: str, err: Exception) -> None:
        if path not in self._warned:
            self._warned.add(path)
            logger.warning("DAGShare RunLogger cannot write %s: %s", path, err)

    def _write(self, name: str, text: str, mode: str) -> None:
        path = os.path.join(self.out_dir, name)
        try:
            with open(path, mode) as f:
                f.write(text)
        except Exception as e:
            self._warn(path, e)

    def _append(self, name: str, record: dict) -> None:
        try:
            line = json.dumps(record, default=str) + "\n"
        except Exception as e:
            self._warn(name, e)
            return
        self._write(name, line, "a")

    def write_config(self, info: dict) -> None:
        try:
            record = {"dagshare": dataclasses.asdict(self.cfg), **info}
            text = json.dumps(record, indent=2, default=str)
        except Exception as e:
            self._warn("config.json", e)
            return
        self._write("config.json", text, "w")

    def maybe_log_ledger(self, now: float, ledger: TenantLedger,
                         stats: dict) -> None:
        if (self._last_ledger_t is not None
                and now - self._last_ledger_t < self._interval_s):
            return
        self._last_ledger_t = now
        try:
            record = {"t": now, "tenants": ledger.snapshot(), **stats}
        except Exception as e:
            self._warn("ledger.jsonl", e)
            return
        self._append("ledger.jsonl", record)

    def event(self, **fields) -> None:
        self._append("events.jsonl", {"t": time.time(), **fields})
