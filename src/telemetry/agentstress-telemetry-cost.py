"""Token pricing and cost tracking for stress test runs.

Tracks per-agent and per-step costs based on configurable token pricing.
Supports multiple model price tiers and computes cost efficiency metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from telemetry.agentstress_telemetry_collect import StepMetrics


# Pricing per 1M tokens (USD) as of early 2026
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "local-vllm": {"input": 0.00, "output": 0.00},
    "default": {"input": 2.50, "output": 10.00},
}


@dataclass
class CostRecord:
    """Cost breakdown for a single step or agent."""

    agent_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class CostTracker:
    """Tracks cumulative token costs across a stress test run.

    Computes per-agent costs, total costs, and cost efficiency metrics
    (cost per successful step, cost overhead from faults, etc.).
    """

    def __init__(
        self,
        model: str = "default",
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self._pricing = pricing or _DEFAULT_PRICING
        self._records: list[CostRecord] = []

    def _get_price(self, direction: str) -> float:
        """Get price per 1M tokens for input or output."""
        model_pricing = self._pricing.get(self.model, self._pricing.get("default", {}))
        return model_pricing.get(direction, 0.0)

    def _compute_cost(self, tokens: int, direction: str) -> float:
        price_per_million = self._get_price(direction)
        return (tokens / 1_000_000) * price_per_million

    def record_step(self, step: StepMetrics) -> CostRecord:
        """Record cost for a single step from telemetry metrics."""
        record = CostRecord(
            agent_id=step.agent_id,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
            input_cost_usd=self._compute_cost(step.input_tokens, "input"),
            output_cost_usd=self._compute_cost(step.output_tokens, "output"),
        )
        self._records.append(record)
        return record

    def record_steps(self, steps: list[StepMetrics]) -> list[CostRecord]:
        return [self.record_step(s) for s in steps]

    # --- Aggregations ---

    @property
    def records(self) -> list[CostRecord]:
        return list(self._records)

    def total_cost(self) -> float:
        return sum(r.total_cost_usd for r in self._records)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    def agent_costs(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        for r in self._records:
            costs[r.agent_id] = costs.get(r.agent_id, 0.0) + r.total_cost_usd
        return costs

    def agent_tokens(self) -> dict[str, int]:
        tokens: dict[str, int] = {}
        for r in self._records:
            tokens[r.agent_id] = tokens.get(r.agent_id, 0) + r.total_tokens
        return tokens

    def cost_per_step(self) -> float:
        if not self._records:
            return 0.0
        return self.total_cost() / len(self._records)

    # --- Cost efficiency ---

    def compute_cost_overhead(
        self, baseline_steps: list[StepMetrics], stressed_steps: list[StepMetrics]
    ) -> dict[str, Any]:
        """Compare cost between baseline and stressed runs.

        Returns overhead ratio and absolute cost difference.
        """
        baseline_tracker = CostTracker(model=self.model, pricing=self._pricing)
        baseline_tracker.record_steps(baseline_steps)

        stressed_tracker = CostTracker(model=self.model, pricing=self._pricing)
        stressed_tracker.record_steps(stressed_steps)

        baseline_cost = baseline_tracker.total_cost()
        stressed_cost = stressed_tracker.total_cost()

        return {
            "baseline_cost_usd": round(baseline_cost, 6),
            "stressed_cost_usd": round(stressed_cost, 6),
            "overhead_usd": round(stressed_cost - baseline_cost, 6),
            "overhead_ratio": round(stressed_cost / baseline_cost, 3) if baseline_cost > 0 else 0.0,
            "baseline_tokens": baseline_tracker.total_tokens(),
            "stressed_tokens": stressed_tracker.total_tokens(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens": self.total_tokens(),
            "total_steps": len(self._records),
            "cost_per_step_usd": round(self.cost_per_step(), 6),
            "agent_costs": {k: round(v, 6) for k, v in self.agent_costs().items()},
            "agent_tokens": self.agent_tokens(),
        }

    def reset(self) -> None:
        self._records.clear()
