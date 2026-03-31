"""Runner package: importlib loader for hyphenated module files."""

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


# Load in dependency order: scenario first (engine imports scenario)
agentstress_runner_scenario = _load(
    "agentstress-runner-scenario.py", "runner.agentstress_runner_scenario"
)
agentstress_runner_engine = _load(
    "agentstress-runner-engine.py", "runner.agentstress_runner_engine"
)
agentstress_runner_report = _load(
    "agentstress-runner-report.py", "runner.agentstress_runner_report"
)
