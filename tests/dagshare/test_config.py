# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


def test_defaults(ds):
    cfg = ds.DagShareConfig.from_env({})
    assert cfg == ds.DagShareConfig()
    assert cfg.compute and cfg.fit_check and not cfg.evict_overserved
    assert cfg.compute_share == "slot"
    assert cfg.alpha == 1.0 and cfg.beta == 1.0
    assert cfg.fit_estimate == "margin" and cfg.fit_margin_blocks == 16
    assert cfg.delta == 0.0 and cfg.charge_mode == "measured"
    assert cfg.log_interval_ms == 500
    assert cfg.run_output_dir == "./results/run"


def test_all_settings(ds):
    cfg = ds.DagShareConfig.from_env({
        "DAGSHARE_COMPUTE": "off",
        "DAGSHARE_COMPUTE_SHARE": "tokens",
        "DAGSHARE_ALPHA": "0",
        "DAGSHARE_BETA": "2.5",
        "DAGSHARE_FIT_CHECK": "OFF",
        "DAGSHARE_FIT_ESTIMATE": "p90",
        "DAGSHARE_FIT_MARGIN": "4",
        "DAGSHARE_DELTA": "0.25",
        "DAGSHARE_CHARGE_MODE": "predicted_corrected",
        "DAGSHARE_EVICT_OVERSERVED": "on",
        "DAGSHARE_LOG_INTERVAL_MS": "100",
        "RUN_OUTPUT_DIR": "/tmp/x",
        "UNRELATED": "ignored",
    })
    assert cfg == ds.DagShareConfig(compute=False,
                                    compute_share="tokens",
                                    alpha=0.0,
                                    beta=2.5,
                                    fit_check=False,
                                    fit_estimate="p90",
                                    fit_margin_blocks=4,
                                    delta=0.25,
                                    charge_mode="predicted_corrected",
                                    evict_overserved=True,
                                    log_interval_ms=100,
                                    run_output_dir="/tmp/x")


def test_empty_value_uses_default(ds):
    cfg = ds.DagShareConfig.from_env({"DAGSHARE_ALPHA": " "})
    assert cfg.alpha == 1.0


@pytest.mark.parametrize("name,value", [
    ("DAGSHARE_COMPUTE", "maybe"),
    ("DAGSHARE_COMPUTE_SHARE", "flops"),
    ("DAGSHARE_ALPHA", "-1"),
    ("DAGSHARE_ALPHA", "abc"),
    ("DAGSHARE_BETA", "nan"),
    ("DAGSHARE_BETA", "inf"),
    ("DAGSHARE_FIT_CHECK", "2"),
    ("DAGSHARE_FIT_ESTIMATE", "p50"),
    ("DAGSHARE_FIT_MARGIN", "1.5"),
    ("DAGSHARE_FIT_MARGIN", "-3"),
    ("DAGSHARE_DELTA", "-0.1"),
    ("DAGSHARE_CHARGE_MODE", "guess"),
    ("DAGSHARE_EVICT_OVERSERVED", "yes please"),
    ("DAGSHARE_LOG_INTERVAL_MS", "fast"),
])
def test_invalid_names_variable(ds, name, value):
    with pytest.raises(ValueError, match=name):
        ds.DagShareConfig.from_env({name: value})
