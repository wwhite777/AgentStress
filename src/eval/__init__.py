"""Eval package: importlib loader for hyphenated module files."""

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


# Load in dependency order
agentstress_eval_judge = _load(
    "agentstress-eval-judge.py", "eval.agentstress_eval_judge"
)
agentstress_eval_score = _load(
    "agentstress-eval-score.py", "eval.agentstress_eval_score"
)
agentstress_eval_blast = _load(
    "agentstress-eval-blast.py", "eval.agentstress_eval_blast"
)
agentstress_eval_compare = _load(
    "agentstress-eval-compare.py", "eval.agentstress_eval_compare"
)
