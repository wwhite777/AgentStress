"""BlastRadiusAnalyzer: kill each agent, measure system-wide impact.

For each agent in the topology, simulates its complete failure and
measures the system's degradation. Outputs a resiliency map showing
which agents are critical vs. redundant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentCriticality(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    REDUNDANT = "redundant"
    UNKNOWN = "unknown"


@dataclass
class AgentBlastResult:
    """Impact of killing a single agent."""

    agent_id: str
    agent_role: str = ""
    baseline_score: float = 0.0
    degraded_score: float = 0.0
    score_delta: float = 0.0
    degradation_pct: float = 0.0
    affected_downstream: list[str] = field(default_factory=list)
    criticality: AgentCriticality = AgentCriticality.UNKNOWN
    cost_change_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "baseline_score": self.baseline_score,
            "degraded_score": self.degraded_score,
            "score_delta": self.score_delta,
            "degradation_pct": self.degradation_pct,
            "affected_downstream": self.affected_downstream,
            "criticality": self.criticality.value,
            "cost_change_pct": self.cost_change_pct,
        }


@dataclass
class BlastRadiusReport:
    """Full blast radius analysis across all agents."""

    topology_name: str = ""
    baseline_score: float = 0.0
    agent_results: list[AgentBlastResult] = field(default_factory=list)

    @property
    def critical_agents(self) -> list[str]:
        return [r.agent_id for r in self.agent_results if r.criticality == AgentCriticality.CRITICAL]

    @property
    def redundant_agents(self) -> list[str]:
        return [r.agent_id for r in self.agent_results if r.criticality == AgentCriticality.REDUNDANT]

    @property
    def most_critical_agent(self) -> str | None:
        if not self.agent_results:
            return None
        worst = max(self.agent_results, key=lambda r: r.degradation_pct)
        return worst.agent_id

    @property
    def system_resilience(self) -> float:
        """Average degradation across all single-agent failures. Lower = more resilient."""
        if not self.agent_results:
            return 0.0
        return round(
            sum(r.degradation_pct for r in self.agent_results) / len(self.agent_results), 2
        )

    def get_result(self, agent_id: str) -> AgentBlastResult | None:
        for r in self.agent_results:
            if r.agent_id == agent_id:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_name": self.topology_name,
            "baseline_score": self.baseline_score,
            "system_resilience_avg_degradation_pct": self.system_resilience,
            "critical_agents": self.critical_agents,
            "redundant_agents": self.redundant_agents,
            "most_critical_agent": self.most_critical_agent,
            "agent_results": [r.to_dict() for r in self.agent_results],
        }


class BlastRadiusAnalyzer:
    """Analyzes blast radius by simulating single-agent failures.

    For each agent:
    1. Kill the agent (skip its LLM calls entirely)
    2. Run the pipeline
    3. Judge the output
    4. Compare against baseline
    5. Classify agent criticality

    Criticality thresholds:
    - CRITICAL: >50% quality degradation
    - IMPORTANT: 20-50% degradation
    - REDUNDANT: <20% degradation
    """

    def __init__(
        self,
        critical_threshold: float = 50.0,
        important_threshold: float = 20.0,
    ) -> None:
        self.critical_threshold = critical_threshold
        self.important_threshold = important_threshold

    def classify_criticality(self, degradation_pct: float) -> AgentCriticality:
        if degradation_pct >= self.critical_threshold:
            return AgentCriticality.CRITICAL
        if degradation_pct >= self.important_threshold:
            return AgentCriticality.IMPORTANT
        return AgentCriticality.REDUNDANT

    def analyze_agent(
        self,
        agent_id: str,
        agent_role: str,
        baseline_score: float,
        degraded_score: float,
        downstream_agents: list[str] | None = None,
        baseline_cost: float = 0.0,
        degraded_cost: float = 0.0,
    ) -> AgentBlastResult:
        """Compute blast radius for a single agent failure."""
        score_delta = round(degraded_score - baseline_score, 4)
        degradation_pct = (
            round((1 - degraded_score / baseline_score) * 100, 2)
            if baseline_score > 0
            else 0.0
        )
        cost_change_pct = (
            round((degraded_cost - baseline_cost) / baseline_cost * 100, 2)
            if baseline_cost > 0
            else 0.0
        )

        return AgentBlastResult(
            agent_id=agent_id,
            agent_role=agent_role,
            baseline_score=baseline_score,
            degraded_score=degraded_score,
            score_delta=score_delta,
            degradation_pct=degradation_pct,
            affected_downstream=downstream_agents or [],
            criticality=self.classify_criticality(degradation_pct),
            cost_change_pct=cost_change_pct,
        )

    def build_report(
        self,
        topology_name: str,
        baseline_score: float,
        agent_results: list[AgentBlastResult],
    ) -> BlastRadiusReport:
        return BlastRadiusReport(
            topology_name=topology_name,
            baseline_score=baseline_score,
            agent_results=agent_results,
        )
