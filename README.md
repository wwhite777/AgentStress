# AgentStress

**Multi-agent reliability testing framework — Jepsen/Chaos Mesh for AI agent teams.**

AgentStress applies Multi-Robot Systems (MRS) fault models to LLM-based multi-agent systems. It provides topology-aware fault injection, deterministic replay, cost-aware degradation curves, and blast radius analysis.

## MRS-to-LLM Fault Mapping

| MRS Concept | AgentStress Fault | Description |
|---|---|---|
| Sensor Noise / Blindness | Context Corruption | Truncate context, inject noise, simulate RAG retrieval failure |
| Packet Loss / Latency | Message Drops / API Throttling | Drop inter-agent messages, simulate rate limits (HTTP 429) |
| Byzantine Faults | Rogue / Hallucinating Agents | Force agents to output confident nonsense or fabricated citations |
| Deadlock / Livelock | Token Thrashing / Infinite Loops | Endless delegation loops, inflated context burning API budget |

## Architecture

```
User Config (YAML) → StressTestEngine → FrameworkAdapter (LangGraph)
                                                           ↓
                      Original Graph nodes get ProxiedChatModel
                                       ↓
                      InterceptionPipeline: [FaultInjectors, Telemetry, Recorder]
                                       ↓
                      Execute baseline (no faults) + stressed (with faults)
                                       ↓
                      LocalJudge (vLLM on L40S) scores outputs
                                       ↓
                      StressTestMetrics + BlastRadius + DegradationCurve → Report
```

See `result/figure/agentstress-system-architecture.drawio.xml.gpg` for the detailed system architecture diagram.

## Quick Start

```bash
pip install -e ".[dev]"
pytest test/
```

196 tests pass covering all modules: topology, proxy, faults (all 4 MRS categories), telemetry, evaluation, runner, replay, and adapter.

## Usage

```bash
# Run a stress test
agentstress run --topology configs/agentstress-topology-example.yaml \
                --scenario configs/agentstress-scenario-context-corrupt.yaml

# Sweep across fault probability levels
agentstress sweep --topology configs/agentstress-topology-example.yaml \
                  --scenario configs/agentstress-scenario-composite.yaml

# Blast radius analysis (kill each agent, measure impact)
agentstress blast --topology configs/agentstress-topology-example.yaml

# Visualize topology as Mermaid
agentstress viz --topology configs/agentstress-topology-example.yaml --format mermaid

# Replay a recorded run
agentstress replay result/recordings/simple-demo.json --step 3
```

## Demo Results

### Degradation Curve (Sweep Mode)

Swept across fault probability levels [20%, 50%, 80%, 100%]:

| Fault Probability | Quality Score | Tokens |
|---|---|---|
| 20% | 0.9500 | 882 |
| 50% | 0.9500 | 1,173 |
| 80% | 0.6000 | 585 |
| 100% | 0.6000 | 582 |

**Resilience Score: 0.8158** — System maintains baseline quality up to 50% fault probability, then drops sharply at 80%.

All evaluation performed locally on L40S with Qwen-32B-AWQ via vLLM — zero external API cost.

## Key Features

- **Topology-aware fault injection**: Target specific agents by ID within defined graph topologies
- **MRS-inspired fault propagation**: Cascading failure analysis borrowed from multi-robot systems
- **9 fault types across 4 MRS categories**: Context truncation/noise/RAG failure, message drop/API throttle, byzantine/hallucination, deadlock/token thrash
- **4 scheduling modes**: Continuous, burst, progressive, once
- **Cost-aware degradation curves**: Accuracy vs. cost efficiency under increasing fault rates
- **Deterministic replay**: Record and replay execution traces with seed-based reproducibility
- **Blast radius analysis**: Kill each agent, measure system-wide impact, classify critical vs. redundant
- **Time-travel debugging**: Step forward/backward through execution traces
- **Local judge evaluation**: vLLM-served Qwen-32B-AWQ on L40S for zero-cost automated scoring
- **JSON + HTML reports**: Styled dashboards with color-coded criticality tables

## Project Structure

```
src/
├── topology/      — AgentNode, AgentEdge, TopologySpec (Pydantic) + YAML loader + Mermaid/Graphviz viz
├── proxy/         — InterceptionPipeline, ProxiedChatModel, ToolProxy, StateProxy
├── faults/        — 9 fault injectors (context, network, byzantine, deadlock) + scheduler
├── telemetry/     — StepMetrics, TelemetryCollector, ExecutionTrace (time-travel), CostTracker
├── eval/          — LocalJudge (vLLM), StressTestMetrics, DegradationCurve, BlastRadiusAnalyzer
├── runner/        — StressTestEngine (orchestrator), ScenarioSpec loader, JSON/HTML reports
├── replay/        — ExecutionRecorder (deterministic), ReplayPlayer (playback + diff)
├── adapters/      — FrameworkAdapter (abstract) + LangGraphAdapter
└── cli/           — CLI: run, sweep, blast, replay, viz
```

## Configuration

Topologies and scenarios are defined in YAML:

```yaml
# configs/agentstress-topology-example.yaml
topology:
  name: medagent-clinical
  agents:
    - id: triage
      role: router
    - id: literature
      role: worker
    - id: reasoning
      role: worker
    - id: report
      role: aggregator
  edges:
    - source: triage
      target: literature
    - source: literature
      target: reasoning
    - source: reasoning
      target: report
```

```yaml
# configs/agentstress-scenario-composite.yaml
scenario:
  name: composite-stress
  faults:
    - fault_type: context_truncation
      probability: 0.5
      target_agents: [reasoning]
      params: { keep_ratio: 0.4 }
    - fault_type: hallucination
      probability: 0.4
      target_agents: [literature]
      params: { num_hallucinations: 1 }
  sweep:
    enabled: true
    values: [0.1, 0.25, 0.5, 0.75, 1.0]
```

## Hardware Requirements

- **Testing (stubs)**: Any machine with Python 3.10+
- **Local judge evaluation**: NVIDIA GPU with 16GB+ VRAM (L40S recommended for Qwen-32B-AWQ with TP=2)
- **Full demo**: L40S or equivalent, vLLM installed

```bash
# Start the judge server
bash scripts/agentstress-serve-judge.sh
```
