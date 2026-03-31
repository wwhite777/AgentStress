"""Adapters package: importlib loader for hyphenated module files."""

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


# Load in dependency order: base first (langgraph imports from base)
agentstress_adapter_base = _load(
    "agentstress-adapter-base.py", "adapters.agentstress_adapter_base"
)
agentstress_adapter_langgraph = _load(
    "agentstress-adapter-langgraph.py", "adapters.agentstress_adapter_langgraph"
)
