"""Proxy package: importlib loader for hyphenated module files."""

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


# Load in dependency order: intercept first (llm imports from intercept)
agentstress_proxy_intercept = _load(
    "agentstress-proxy-intercept.py", "proxy.agentstress_proxy_intercept"
)
agentstress_proxy_llm = _load(
    "agentstress-proxy-llm.py", "proxy.agentstress_proxy_llm"
)
agentstress_proxy_tool = _load(
    "agentstress-proxy-tool.py", "proxy.agentstress_proxy_tool"
)
agentstress_proxy_message = _load(
    "agentstress-proxy-message.py", "proxy.agentstress_proxy_message"
)
