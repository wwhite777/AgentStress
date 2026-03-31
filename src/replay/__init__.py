"""Replay package: importlib loader for hyphenated module files."""

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


# Load in dependency order: record first (player imports from record)
agentstress_replay_record = _load(
    "agentstress-replay-record.py", "replay.agentstress_replay_record"
)
agentstress_replay_player = _load(
    "agentstress-replay-player.py", "replay.agentstress_replay_player"
)
