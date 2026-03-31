#!/usr/bin/env bash
# Run a stress test scenario against a topology.
# Usage: bash scripts/agentstress-run-stress.sh [--scenario SCENARIO] [--topology TOPOLOGY]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TOPOLOGY="${TOPOLOGY:-${PROJECT_DIR}/configs/agentstress-topology-example.yaml}"
SCENARIO="${SCENARIO:-${PROJECT_DIR}/configs/agentstress-scenario-context-corrupt.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/result}"
MODEL="${MODEL:-default}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --topology|-t) TOPOLOGY="$2"; shift 2 ;;
        --scenario|-s) SCENARIO="$2"; shift 2 ;;
        --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --sweep) COMMAND="sweep"; shift ;;
        --blast) COMMAND="blast"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

COMMAND="${COMMAND:-run}"

echo "=== AgentStress ==="
echo "Command:   ${COMMAND}"
echo "Topology:  ${TOPOLOGY}"
echo "Scenario:  ${SCENARIO}"
echo "Output:    ${OUTPUT_DIR}"
echo "==================="

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python3 "${PROJECT_DIR}/src/cli/agentstress-cli-main.py" \
    "${COMMAND}" \
    --topology "${TOPOLOGY}" \
    --scenario "${SCENARIO}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --no-judge
