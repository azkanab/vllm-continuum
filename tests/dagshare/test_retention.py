# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

FIXED_THRESHOLD_CONTINUUM = 2.0


def continuum_set_up_pin(this_func_call, tool_avg_s):
    """Copy of ToolCallEstimator.set_up_pin in vllm/v1/core/estimate_with_func.py."""
    if this_func_call is None:
        return 0
    this_func_call_exec_time = tool_avg_s or 0.0
    if this_func_call_exec_time > FIXED_THRESHOLD_CONTINUUM:
        return 0
    return FIXED_THRESHOLD_CONTINUUM


@pytest.mark.parametrize("policy,kind,seconds", [
    ("continuum", "ttl", 2.0),
    ("dagshare", "ttl", 2.0),
    ("DAGSHARE", "ttl", 2.0),
    ("fcfs", "none", 0.0),
    ("priority", "none", 0.0),
])
def test_default_per_policy(ds, policy, kind, seconds):
    rule = ds.RetentionRule.from_env(policy, {})
    assert (rule.kind, rule.seconds, rule.fast_tools_only) == (kind, seconds,
                                                               True)


@pytest.mark.parametrize("value,expected", [
    ("none", ("none", 0.0)),
    ("ttl:2", ("ttl", 2.0)),
    ("ttl:0.5", ("ttl", 0.5)),
    ("until_done:600", ("until_done", 600.0)),
    ("TTL:3", ("ttl", 3.0)),
])
def test_parse(ds, value, expected):
    rule = ds.RetentionRule.from_env("fcfs", {"KV_RETENTION": value})
    assert (rule.kind, rule.seconds) == expected


def test_fast_tools_only_flag(ds):
    rule = ds.RetentionRule.from_env(
        "continuum", {"KV_RETENTION_FAST_TOOLS_ONLY": "false"})
    assert not rule.fast_tools_only


@pytest.mark.parametrize("env", [
    {"KV_RETENTION": "ttl"},
    {"KV_RETENTION": "ttl:"},
    {"KV_RETENTION": "ttl:-1"},
    {"KV_RETENTION": "until_done:x"},
    {"KV_RETENTION": "forever"},
    {"KV_RETENTION": "none:3"},
    {"KV_RETENTION_FAST_TOOLS_ONLY": "sometimes"},
])
def test_invalid(ds, env):
    with pytest.raises(ValueError, match=next(iter(env))):
        ds.RetentionRule.from_env("continuum", env)


@pytest.mark.parametrize("this_func_call", [None, "bash", "search"])
@pytest.mark.parametrize("tool_avg_s",
                         [None, 0.0, 0.5, 1.999, 2.0, 2.0001, 30.0])
@pytest.mark.parametrize("is_last_step", [None, False])
def test_ttl2_reproduces_continuum(ds, this_func_call, tool_avg_s,
                                   is_last_step):
    rule = ds.RetentionRule.from_env("continuum", {})
    got = rule.pin_seconds(this_func_call=this_func_call,
                           tool_avg_s=tool_avg_s,
                           is_last_step=is_last_step)
    assert got == continuum_set_up_pin(this_func_call, tool_avg_s)


@pytest.mark.parametrize("kind,seconds", [("none", 0.0), ("ttl", 2.0),
                                          ("until_done", 600.0)])
def test_last_step_never_pins(ds, kind, seconds):
    rule = ds.RetentionRule(kind, seconds)
    assert rule.pin_seconds(this_func_call="bash",
                            tool_avg_s=0.1,
                            is_last_step=True) == 0


def test_none_never_pins(ds):
    rule = ds.RetentionRule("none")
    assert rule.pin_seconds(this_func_call="bash",
                            tool_avg_s=0.0,
                            is_last_step=False) == 0


def test_ttl_slow_tools_when_not_fast_only(ds):
    rule = ds.RetentionRule("ttl", 2.0, fast_tools_only=False)
    assert rule.pin_seconds(this_func_call="bash",
                            tool_avg_s=30.0,
                            is_last_step=False) == 2.0
    assert rule.pin_seconds(this_func_call=None,
                            tool_avg_s=0.0,
                            is_last_step=False) == 0


@pytest.mark.parametrize("tool_avg_s", [None, 0.5, 100.0])
def test_until_done(ds, tool_avg_s):
    rule = ds.RetentionRule("until_done", 600.0)
    assert rule.pin_seconds(this_func_call="bash",
                            tool_avg_s=tool_avg_s,
                            is_last_step=False) == 600.0
    assert rule.pin_seconds(this_func_call=None,
                            tool_avg_s=tool_avg_s,
                            is_last_step=False) == 0
