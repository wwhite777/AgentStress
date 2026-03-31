"""Tests for CostTracker."""

from __future__ import annotations

import pytest

from telemetry.agentstress_telemetry_collect import StepMetrics
from telemetry.agentstress_telemetry_cost import CostRecord, CostTracker


def _make_step(agent_id: str, input_tokens: int, output_tokens: int) -> StepMetrics:
    return StepMetrics(
        agent_id=agent_id,
        step_index=1,
        timestamp=0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestAgentStressCostRecord:
    """Tests for CostRecord dataclass."""

    def test_agentstress_cost_record_totals(self):
        record = CostRecord(
            agent_id="test",
            input_tokens=1000,
            output_tokens=500,
            input_cost_usd=0.0025,
            output_cost_usd=0.005,
        )
        assert record.total_tokens == 1500
        assert record.total_cost_usd == pytest.approx(0.0075)


class TestAgentStressCostTracker:
    """Tests for CostTracker."""

    def test_agentstress_cost_tracker_record_step(self):
        tracker = CostTracker(model="gpt-4o")
        step = _make_step("reasoning", input_tokens=1000, output_tokens=500)
        record = tracker.record_step(step)

        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.input_cost_usd > 0
        assert record.output_cost_usd > 0

    def test_agentstress_cost_tracker_gpt4o_pricing(self):
        tracker = CostTracker(model="gpt-4o")
        step = _make_step("test", input_tokens=1_000_000, output_tokens=1_000_000)
        record = tracker.record_step(step)

        assert record.input_cost_usd == pytest.approx(2.50)
        assert record.output_cost_usd == pytest.approx(10.00)

    def test_agentstress_cost_tracker_local_vllm_free(self):
        tracker = CostTracker(model="local-vllm")
        step = _make_step("test", input_tokens=1_000_000, output_tokens=1_000_000)
        record = tracker.record_step(step)

        assert record.total_cost_usd == 0.0

    def test_agentstress_cost_tracker_agent_costs(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record_step(_make_step("triage", 500, 200))
        tracker.record_step(_make_step("reasoning", 1000, 800))
        tracker.record_step(_make_step("reasoning", 600, 400))

        agent_costs = tracker.agent_costs()
        assert "triage" in agent_costs
        assert "reasoning" in agent_costs
        assert agent_costs["reasoning"] > agent_costs["triage"]

    def test_agentstress_cost_tracker_total_cost(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record_step(_make_step("a", 1000, 500))
        tracker.record_step(_make_step("b", 2000, 1000))

        assert tracker.total_cost() > 0
        assert tracker.total_tokens() == 4500

    def test_agentstress_cost_tracker_cost_per_step(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record_step(_make_step("a", 1000, 500))
        tracker.record_step(_make_step("b", 1000, 500))

        assert tracker.cost_per_step() == pytest.approx(tracker.total_cost() / 2)

    def test_agentstress_cost_tracker_cost_per_step_empty(self):
        tracker = CostTracker()
        assert tracker.cost_per_step() == 0.0

    def test_agentstress_cost_tracker_cost_overhead(self):
        tracker = CostTracker(model="gpt-4o")

        baseline = [_make_step("a", 1000, 500)]
        stressed = [_make_step("a", 3000, 1500)]  # 3x more tokens due to faults

        result = tracker.compute_cost_overhead(baseline, stressed)

        assert result["overhead_ratio"] == pytest.approx(3.0)
        assert result["overhead_usd"] > 0
        assert result["stressed_tokens"] == 3 * result["baseline_tokens"]

    def test_agentstress_cost_tracker_summary(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record_step(_make_step("triage", 500, 200))
        tracker.record_step(_make_step("reasoning", 1000, 800))

        summary = tracker.summary()
        assert summary["model"] == "gpt-4o"
        assert summary["total_steps"] == 2
        assert summary["total_tokens"] == 2500
        assert "triage" in summary["agent_costs"]
        assert "reasoning" in summary["agent_tokens"]

    def test_agentstress_cost_tracker_custom_pricing(self):
        custom = {"my-model": {"input": 1.0, "output": 2.0}}
        tracker = CostTracker(model="my-model", pricing=custom)
        step = _make_step("test", input_tokens=1_000_000, output_tokens=1_000_000)
        record = tracker.record_step(step)

        assert record.input_cost_usd == pytest.approx(1.0)
        assert record.output_cost_usd == pytest.approx(2.0)

    def test_agentstress_cost_tracker_reset(self):
        tracker = CostTracker(model="gpt-4o")
        tracker.record_step(_make_step("a", 1000, 500))
        assert len(tracker.records) == 1

        tracker.reset()
        assert len(tracker.records) == 0
        assert tracker.total_cost() == 0.0
