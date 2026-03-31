"""Simple example: 3-agent linear topology with context truncation fault.

Demonstrates the core AgentStress workflow without requiring a real LLM
or a running graph framework — uses stub models for everything.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import topology  # noqa: F401
import proxy  # noqa: F401
import faults  # noqa: F401
import telemetry  # noqa: F401
import adapters  # noqa: F401
import eval  # noqa: F401
import runner  # noqa: F401
import replay  # noqa: F401

from faults.agentstress_fault_base import FaultConfig, FaultType
from faults.agentstress_fault_context import ContextTruncationFault
from proxy.agentstress_proxy_intercept import InterceptionContext, InterceptionPipeline
from proxy.agentstress_proxy_llm import ProxiedChatModel
from telemetry.agentstress_telemetry_collect import TelemetryCollector
from telemetry.agentstress_telemetry_cost import CostTracker
from topology.agentstress_topology_parse import build_linear_topology
from replay.agentstress_replay_record import ExecutionRecorder

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any


class DemoAgent(BaseChatModel):
    """Simple deterministic agent for demo purposes."""

    agent_name: str = "agent"

    @property
    def _llm_type(self) -> str:
        return "demo"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        last_msg = messages[-1].content if messages else "nothing"
        response = f"[{self.agent_name}] Processed: {last_msg[:80]}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])


def main():
    print("=" * 60)
    print("AgentStress Simple Example")
    print("=" * 60)

    # 1. Define topology
    topo = build_linear_topology(["intake", "analyzer", "summarizer"])
    print(f"\nTopology: {topo.name}")
    print(f"  Agents: {topo.get_agent_ids()}")
    print(f"  Edges: {[(e.source, e.target) for e in topo.edges]}")

    # 2. Set up pipeline with fault + telemetry + recorder
    pipeline = InterceptionPipeline()

    fault_config = FaultConfig(
        fault_type=FaultType.CONTEXT_TRUNCATION,
        probability=1.0,
        target_agents=["analyzer"],
        params={"keep_ratio": 0.3, "keep_system": True},
    )
    fault = ContextTruncationFault(fault_config)
    telemetry_collector = TelemetryCollector()
    recorder = ExecutionRecorder(run_id="simple-demo", master_seed=42)

    pipeline.add(fault)
    pipeline.add(telemetry_collector)
    pipeline.add(recorder)

    # 3. Create proxied agents
    agents = {}
    for agent_id in topo.get_agent_ids():
        base_model = DemoAgent(agent_name=agent_id)
        agents[agent_id] = ProxiedChatModel(
            wrapped=base_model,
            pipeline=pipeline,
            agent_id=agent_id,
        )

    # 4. Simulate a multi-agent run
    print("\n--- Baseline (no faults) ---")
    pipeline.remove(fault.name)
    messages = [HumanMessage(content="Patient: 45M with chest pain, dyspnea, elevated troponin.")]
    for agent_id in topo.get_agent_ids():
        result = agents[agent_id].invoke(messages)
        messages.append(result)
        print(f"  {agent_id}: {result.content}")

    print("\n--- Stressed (context truncation on analyzer) ---")
    pipeline.clear()
    pipeline.add(fault)
    pipeline.add(telemetry_collector)
    pipeline.add(recorder)

    messages = [HumanMessage(content="Patient: 45M with chest pain, dyspnea, elevated troponin.")]
    for agent_id in topo.get_agent_ids():
        result = agents[agent_id].invoke(messages)
        messages.append(result)
        print(f"  {agent_id}: {result.content}")

    # 5. Print telemetry
    print("\n--- Telemetry ---")
    summary = telemetry_collector.summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Faults triggered: {summary['faults_triggered']}")

    # 6. Print cost
    cost_tracker = CostTracker(model="gpt-4o")
    cost_tracker.record_steps(telemetry_collector.steps)
    cost_summary = cost_tracker.summary()
    print(f"\n--- Cost ---")
    print(f"  Total cost: ${cost_summary['total_cost_usd']:.6f}")
    print(f"  Per agent: {cost_summary['agent_costs']}")

    # 7. Save recording
    rec_path = Path("result/recordings/simple-demo.json")
    recorder.save(rec_path)
    print(f"\n--- Recording saved to {rec_path} ---")

    # 8. Replay
    from replay.agentstress_replay_player import ReplayPlayer

    recording = ExecutionRecorder.load(rec_path)
    player = ReplayPlayer(recording)
    print(f"\n--- Replay ({player.total_steps} steps) ---")
    print(json.dumps(player.summary(), indent=2))

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
