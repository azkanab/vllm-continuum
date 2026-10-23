# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_NAME = "_dagshare_under_test"
_MODULE_PATH = (Path(__file__).resolve().parents[2] / "vllm" / "v1" / "core" /
                "sched" / "dagshare.py")


def _load_dagshare():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ds():
    """The dagshare module, loaded by path so vllm (and torch) never import."""
    return _load_dagshare()
