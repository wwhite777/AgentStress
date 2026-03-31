#!/usr/bin/env bash
# Run the MedAgent demo with AgentStress.
# Usage: bash scripts/agentstress-run-demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== AgentStress MedAgent Demo ==="

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python3 "${PROJECT_DIR}/examples/agentstress-example-medagent.py"
