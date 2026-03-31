"""Telemetry package: importlib loader for hyphenated module files."""

import importlib.util
import sys
from pathlib import Path

_dir = Path(__file__).parent


def _load(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, _dir / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


agentstress_telemetry_collect = _load(
    "agentstress-telemetry-collect.py", "telemetry.agentstress_telemetry_collect"
)
agentstress_telemetry_trace = _load(
    "agentstress-telemetry-trace.py", "telemetry.agentstress_telemetry_trace"
)
agentstress_telemetry_cost = _load(
    "agentstress-telemetry-cost.py", "telemetry.agentstress_telemetry_cost"
)
