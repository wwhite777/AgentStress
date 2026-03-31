"""Baseline vs stressed comparison: structured diff of two test runs.

Compares quality, cost, latency, and per-agent metrics between a
baseline (no faults) and stressed (fault-injected) run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from telemetry.agentstress_telemetry_collect import TelemetryCollector
from telemetry.agentstress_telemetry_cost import CostTracker


@dataclass
class AgentComparison:
    """Per-agent comparison between baseline and stressed runs."""

    agent_id: str
    baseline_tokens: int = 0
    stressed_tokens: int = 0
    baseline_cost_usd: float = 0.0
    stressed_cost_usd: float = 0.0
    baseline_steps: int = 0
    stressed_steps: int = 0
    faults_received: int = 0
    token_overhead_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "baseline_tokens": self.baseline_tokens,
            "stressed_tokens": self.stressed_tokens,
            "token_overhead_pct": self.token_overhead_pct,
            "baseline_cost_usd": round(self.baseline_cost_usd, 6),
            "stressed_cost_usd": round(self.stressed_cost_usd, 6),
            "faults_received": self.faults_received,
        }


@dataclass
class ComparisonReport:
    """Full comparison between baseline and stressed runs."""

    scenario_name: str = ""

    # Overall scores
    baseline_score: float = 0.0
    stressed_score: float = 0.0
    quality_delta: float = 0.0
    degradation_pct: float = 0.0

    # Overall cost
    baseline_total_cost: float = 0.0
    stressed_total_cost: float = 0.0
    cost_overhead_ratio: float = 1.0

    # Overall tokens
    baseline_total_tokens: int = 0
    stressed_total_tokens: int = 0

    # Overall latency
    baseline_total_latency_ms: float = 0.0
    stressed_total_latency_ms: float = 0.0

    # Per-agent breakdowns
    agent_comparisons: list[AgentComparison] = field(default_factory=list)

    # Fault summary
    total_faults_triggered: int = 0

    @property
    def most_affected_agent(self) -> str | None:
        if not self.agent_comparisons:
            return None
        faulted = [a for a in self.agent_comparisons if a.faults_received > 0]
        if not faulted:
            return None
        return max(faulted, key=lambda a: a.token_overhead_pct).agent_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "quality": {
                "baseline_score": self.baseline_score,
                "stressed_score": self.stressed_score,
                "quality_delta": self.quality_delta,
                "degradation_pct": self.degradation_pct,
            },
            "cost": {
                "baseline_usd": round(self.baseline_total_cost, 6),
                "stressed_usd": round(self.stressed_total_cost, 6),
                "overhead_ratio": self.cost_overhead_ratio,
            },
            "tokens": {
                "baseline": self.baseline_total_tokens,
                "stressed": self.stressed_total_tokens,
            },
            "latency": {
                "baseline_ms": round(self.baseline_total_latency_ms, 2),
                "stressed_ms": round(self.stressed_total_latency_ms, 2),
            },
            "faults_triggered": self.total_faults_triggered,
            "most_affected_agent": self.most_affected_agent,
            "agent_comparisons": [a.to_dict() for a in self.agent_comparisons],
        }


def compare_runs(
    scenario_name: str,
    baseline_score: float,
    stressed_score: float,
    baseline_telemetry: TelemetryCollector,
    stressed_telemetry: TelemetryCollector,
    baseline_cost: CostTracker | None = None,
    stressed_cost: CostTracker | None = None,
) -> ComparisonReport:
    """Build a ComparisonReport from telemetry and cost data of two runs."""
    report = ComparisonReport(scenario_name=scenario_name)

    report.baseline_score = baseline_score
    report.stressed_score = stressed_score
    report.quality_delta = round(stressed_score - baseline_score, 4)
    if baseline_score > 0:
        report.degradation_pct = round((1 - stressed_score / baseline_score) * 100, 2)

    b_summary = baseline_telemetry.summary()
    s_summary = stressed_telemetry.summary()

    report.baseline_total_latency_ms = b_summary["total_latency_ms"]
    report.stressed_total_latency_ms = s_summary["total_latency_ms"]
    report.baseline_total_tokens = b_summary["total_tokens"]
    report.stressed_total_tokens = s_summary["total_tokens"]
    report.total_faults_triggered = s_summary["faults_triggered"]

    if baseline_cost:
        report.baseline_total_cost = baseline_cost.total_cost()
    if stressed_cost:
        report.stressed_total_cost = stressed_cost.total_cost()
    if report.baseline_total_cost > 0:
        report.cost_overhead_ratio = round(
            report.stressed_total_cost / report.baseline_total_cost, 3
        )

    all_agents = set(b_summary["agents_observed"]) | set(s_summary["agents_observed"])
    for agent_id in sorted(all_agents):
        b_steps = baseline_telemetry.get_agent_steps(agent_id)
        s_steps = stressed_telemetry.get_agent_steps(agent_id)

        b_tokens = sum(s.total_tokens for s in b_steps)
        s_tokens = sum(s.total_tokens for s in s_steps)
        faults = sum(1 for s in s_steps if s.fault_applied is not None)

        b_cost = 0.0
        s_cost = 0.0
        if baseline_cost:
            b_cost = baseline_cost.agent_costs().get(agent_id, 0.0)
        if stressed_cost:
            s_cost = stressed_cost.agent_costs().get(agent_id, 0.0)

        overhead_pct = round((s_tokens - b_tokens) / b_tokens * 100, 2) if b_tokens > 0 else 0.0

        report.agent_comparisons.append(
            AgentComparison(
                agent_id=agent_id,
                baseline_tokens=b_tokens,
                stressed_tokens=s_tokens,
                baseline_cost_usd=b_cost,
                stressed_cost_usd=s_cost,
                baseline_steps=len(b_steps),
                stressed_steps=len(s_steps),
                faults_received=faults,
                token_overhead_pct=overhead_pct,
            )
        )

    return report
