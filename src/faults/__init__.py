"""Faults package: importlib loader for hyphenated module files."""

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


# Load in dependency order: base first (context imports from base)
agentstress_fault_base = _load(
    "agentstress-fault-base.py", "faults.agentstress_fault_base"
)
agentstress_fault_context = _load(
    "agentstress-fault-context.py", "faults.agentstress_fault_context"
)
agentstress_fault_network = _load(
    "agentstress-fault-network.py", "faults.agentstress_fault_network"
)
agentstress_fault_byzantine = _load(
    "agentstress-fault-byzantine.py", "faults.agentstress_fault_byzantine"
)
agentstress_fault_deadlock = _load(
    "agentstress-fault-deadlock.py", "faults.agentstress_fault_deadlock"
)
agentstress_fault_schedule = _load(
    "agentstress-fault-schedule.py", "faults.agentstress_fault_schedule"
)
