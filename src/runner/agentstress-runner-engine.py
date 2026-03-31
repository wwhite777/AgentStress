"""StressTestEngine: main orchestrator for running stress tests.

Pipeline: config → adapt → run baseline + stressed → evaluate → report.
Supports single runs, sweep mode, and blast radius analysis.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from adapters.agentstress_adapter_base import FrameworkAdapter
from adapters.agentstress_adapter_langgraph import LangGraphAdapter
from eval.agentstress_eval_blast import AgentBlastResult, BlastRadiusAnalyzer, BlastRadiusReport
from eval.agentstress_eval_compare import ComparisonReport, compare_runs
from eval.agentstress_eval_judge import JudgeResult, LocalJudge
from eval.agentstress_eval_score import DegradationCurve, StressTestMetrics, compute_metrics
from faults.agentstress_fault_base import FaultConfig, FaultType
from proxy.agentstress_proxy_intercept import InterceptionPipeline
from runner.agentstress_runner_scenario import ScenarioSpec, instantiate_faults
from telemetry.agentstress_telemetry_collect import TelemetryCollector
from telemetry.agentstress_telemetry_cost import CostTracker
from telemetry.agentstress_telemetry_trace import ExecutionTrace
from topology.agentstress_topology_define import TopologySpec


@dataclass
class RunResult:
    """Result of a single stress test run (baseline or stressed)."""

    output: dict[str, Any] = field(default_factory=dict)
    output_text: str = ""
    telemetry: TelemetryCollector = field(default_factory=TelemetryCollector)
    cost: CostTracker = field(default_factory=CostTracker)
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)
    judge_result: JudgeResult | None = None
    duration_ms: float = 0.0


@dataclass
class StressTestResult:
    """Complete result of a stress test (baseline + stressed + comparison)."""

    scenario_name: str = ""
    topology_name: str = ""
    baseline: RunResult = field(default_factory=RunResult)
    stressed: RunResult = field(default_factory=RunResult)
    metrics: StressTestMetrics | None = None
    comparison: ComparisonReport | None = None
    degradation_curve: DegradationCurve | None = None
    blast_radius: BlastRadiusReport | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "topology_name": self.topology_name,
            "baseline_score": self.baseline.judge_result.overall_score if self.baseline.judge_result else None,
            "stressed_score": self.stressed.judge_result.overall_score if self.stressed.judge_result else None,
        }
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        if self.comparison:
            result["comparison"] = self.comparison.to_dict()
        if self.degradation_curve:
            result["degradation_curve"] = self.degradation_curve.to_dict()
        if self.blast_radius:
            result["blast_radius"] = self.blast_radius.to_dict()
        return result


class StressTestEngine:
    """Main orchestrator for AgentStress test runs.

    Usage:
        engine = StressTestEngine(topology, scenario)
        result = await engine.run(app, inputs)
    """

    def __init__(
        self,
        topology: TopologySpec,
        scenario: ScenarioSpec,
        model: str = "default",
        judge_enabled: bool = True,
    ) -> None:
        self.topology = topology
        self.scenario = scenario
        self.model = model
        self.judge_enabled = judge_enabled

        self._pipeline = InterceptionPipeline()
        self._adapter = LangGraphAdapter(topology, self._pipeline)
        self._judge: LocalJudge | None = None

        if judge_enabled:
            self._judge = LocalJudge(
                base_url=scenario.evaluation.judge_base_url,
                model=scenario.evaluation.judge_model,
            )

    async def _execute_run(
        self,
        app: Any,
        inputs: dict[str, Any],
        faults: list | None = None,
        run_id: str = "",
    ) -> RunResult:
        """Execute a single run (baseline or stressed)."""
        result = RunResult(
            telemetry=TelemetryCollector(),
            cost=CostTracker(model=self.model),
            trace=ExecutionTrace(run_id=run_id),
        )

        self._pipeline.clear()
        if faults:
            for fault in faults:
                self._pipeline.add(fault)
        self._pipeline.add(result.telemetry)

        start = time.time()
        try:
            output = await self._adapter.run(app, inputs)
            result.output = output if isinstance(output, dict) else {}
            result.output_text = self._extract_output_text(output)
        except Exception as e:
            result.output_text = f"[ERROR] {e}"
        result.duration_ms = (time.time() - start) * 1000

        result.cost.record_steps(result.telemetry.steps)

        return result

    def _extract_output_text(self, output: Any) -> str:
        """Extract readable text from graph output for judge evaluation."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ["soap_note", "report", "output", "result", "response"]:
                if key in output and isinstance(output[key], str):
                    return output[key]
            return str(output)
        return str(output)

    async def _judge_output(self, output_text: str, expected: str = "") -> JudgeResult | None:
        if not self._judge or not self.judge_enabled:
            return None
        return self._judge.evaluate(output_text, expected)

    async def run(
        self,
        app: Any,
        inputs: dict[str, Any],
        expected_output: str = "",
    ) -> StressTestResult:
        """Run a complete stress test: baseline + stressed + comparison."""
        test_result = StressTestResult(
            scenario_name=self.scenario.name,
            topology_name=self.topology.name,
        )

        self._adapter.wrap(app)

        test_result.baseline = await self._execute_run(
            app, inputs, faults=None, run_id=f"{self.scenario.name}-baseline"
        )
        test_result.baseline.judge_result = await self._judge_output(
            test_result.baseline.output_text, expected_output
        )

        faults = instantiate_faults(self.scenario)
        test_result.stressed = await self._execute_run(
            app, inputs, faults=faults, run_id=f"{self.scenario.name}-stressed"
        )
        test_result.stressed.judge_result = await self._judge_output(
            test_result.stressed.output_text, expected_output
        )

        baseline_score = test_result.baseline.judge_result.overall_score if test_result.baseline.judge_result else 0.0
        stressed_score = test_result.stressed.judge_result.overall_score if test_result.stressed.judge_result else 0.0

        cost_overhead = test_result.stressed.cost.compute_cost_overhead(
            test_result.baseline.telemetry.steps,
            test_result.stressed.telemetry.steps,
        )

        fault_type_str = self.scenario.fault_configs[0].fault_type.value if self.scenario.fault_configs else "none"
        fault_prob = self.scenario.fault_configs[0].probability if self.scenario.fault_configs else 0.0

        test_result.metrics = compute_metrics(
            scenario_name=self.scenario.name,
            fault_type=fault_type_str,
            fault_probability=fault_prob,
            baseline_judge_result=test_result.baseline.judge_result,
            stressed_judge_result=test_result.stressed.judge_result,
            baseline_telemetry_summary=test_result.baseline.telemetry.summary(),
            stressed_telemetry_summary=test_result.stressed.telemetry.summary(),
            cost_overhead=cost_overhead,
        )

        test_result.comparison = compare_runs(
            scenario_name=self.scenario.name,
            baseline_score=baseline_score,
            stressed_score=stressed_score,
            baseline_telemetry=test_result.baseline.telemetry,
            stressed_telemetry=test_result.stressed.telemetry,
            baseline_cost=test_result.baseline.cost,
            stressed_cost=test_result.stressed.cost,
        )

        return test_result

    async def run_sweep(
        self,
        app: Any,
        inputs: dict[str, Any],
        expected_output: str = "",
    ) -> StressTestResult:
        """Run sweep mode: same scenario at multiple fault probability levels."""
        if not self.scenario.sweep.enabled or not self.scenario.sweep.values:
            return await self.run(app, inputs, expected_output)

        test_result = StressTestResult(
            scenario_name=self.scenario.name,
            topology_name=self.topology.name,
        )

        self._adapter.wrap(app)

        test_result.baseline = await self._execute_run(
            app, inputs, faults=None, run_id=f"{self.scenario.name}-baseline"
        )
        test_result.baseline.judge_result = await self._judge_output(
            test_result.baseline.output_text, expected_output
        )

        baseline_score = test_result.baseline.judge_result.overall_score if test_result.baseline.judge_result else 0.0
        baseline_cost = test_result.baseline.cost.total_cost()

        fault_type_str = self.scenario.fault_configs[0].fault_type.value if self.scenario.fault_configs else "none"

        curve = DegradationCurve(
            scenario_name=self.scenario.name,
            fault_type=fault_type_str,
            baseline_score=baseline_score,
            baseline_cost_usd=baseline_cost,
        )

        for prob in self.scenario.sweep.values:
            faults = instantiate_faults(self.scenario, probability_override=prob)
            run = await self._execute_run(
                app, inputs, faults=faults, run_id=f"{self.scenario.name}-sweep-{prob}"
            )
            run.judge_result = await self._judge_output(run.output_text, expected_output)

            score = run.judge_result.overall_score if run.judge_result else 0.0
            cost = run.cost.total_cost()

            curve.add_point(
                fault_probability=prob,
                quality_score=score,
                cost_usd=cost,
                tokens=run.telemetry.get_total_tokens(),
                latency_ms=run.telemetry.get_total_latency_ms(),
            )

        test_result.degradation_curve = curve
        test_result.stressed = run  # last sweep run

        return test_result

    async def run_blast_radius(
        self,
        app: Any,
        inputs: dict[str, Any],
        expected_output: str = "",
    ) -> StressTestResult:
        """Run blast radius analysis: kill each agent, measure impact."""
        test_result = StressTestResult(
            scenario_name="blast-radius",
            topology_name=self.topology.name,
        )

        self._adapter.wrap(app)

        test_result.baseline = await self._execute_run(
            app, inputs, faults=None, run_id="blast-baseline"
        )
        test_result.baseline.judge_result = await self._judge_output(
            test_result.baseline.output_text, expected_output
        )

        baseline_score = test_result.baseline.judge_result.overall_score if test_result.baseline.judge_result else 0.0
        baseline_cost = test_result.baseline.cost.total_cost()

        analyzer = BlastRadiusAnalyzer()
        agent_results = []

        for agent in self.topology.agents:
            kill_config = FaultConfig(
                fault_type=FaultType.API_THROTTLE,
                probability=1.0,
                target_agents=[agent.id],
                params={"mode": "error", "error_message": f"[KILLED] Agent {agent.id} is offline."},
            )
            from faults.agentstress_fault_network import APIThrottleFault

            kill_fault = APIThrottleFault(kill_config)

            run = await self._execute_run(
                app, inputs, faults=[kill_fault], run_id=f"blast-kill-{agent.id}"
            )
            run.judge_result = await self._judge_output(run.output_text, expected_output)

            degraded_score = run.judge_result.overall_score if run.judge_result else 0.0
            degraded_cost = run.cost.total_cost()

            downstream = self.topology.get_neighbors(agent.id)
            result = analyzer.analyze_agent(
                agent_id=agent.id,
                agent_role=agent.role.value,
                baseline_score=baseline_score,
                degraded_score=degraded_score,
                downstream_agents=downstream,
                baseline_cost=baseline_cost,
                degraded_cost=degraded_cost,
            )
            agent_results.append(result)

        test_result.blast_radius = analyzer.build_report(
            topology_name=self.topology.name,
            baseline_score=baseline_score,
            agent_results=agent_results,
        )

        return test_result

    def close(self) -> None:
        if self._judge:
            self._judge.close()
