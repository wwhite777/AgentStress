"""Topology package: importlib loader for hyphenated module files."""

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


# Load in dependency order: define first (parse imports from define)
agentstress_topology_define = _load(
    "agentstress-topology-define.py", "topology.agentstress_topology_define"
)
agentstress_topology_parse = _load(
    "agentstress-topology-parse.py", "topology.agentstress_topology_parse"
)
agentstress_topology_visualize = _load(
    "agentstress-topology-visualize.py", "topology.agentstress_topology_visualize"
)
