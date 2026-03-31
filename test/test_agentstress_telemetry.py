"""Tests for TelemetryCollector."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from proxy.agentstress_proxy_intercept import InterceptionContext
from telemetry.agentstress_telemetry_collect import StepMetrics, TelemetryCollector


class TestAgentStressStepMetrics:
    """Tests for StepMetrics dataclass."""

    def test_agentstress_telemetry_step_total_tokens(self):
        step = StepMetrics(
            agent_id="test",
            step_index=1,
            timestamp=0.0,
            input_tokens=100,
            output_tokens=50,
        )
        assert step.total_tokens == 150


class TestAgentStressTelemetryCollector:
    """Tests for the TelemetryCollector interceptor."""

    def test_agentstress_telemetry_collector_records_step(self):
        collector = TelemetryCollector()
        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Test input message for token counting")],
            step_index=1,
        )

        ctx = collector.before_call(ctx)
        ctx.response = AIMessage(content="Test response")
        ctx = collector.after_call(ctx)

        assert len(collector.steps) == 1
        step = collector.steps[0]
        assert step.agent_id == "reasoning"
        assert step.step_index == 1
        assert step.input_tokens > 0
        assert step.output_tokens > 0
        assert step.latency_ms >= 0

    def test_agentstress_telemetry_collector_multiple_agents(self):
        collector = TelemetryCollector()

        for agent_id in ["triage", "reasoning", "report"]:
            ctx = InterceptionContext(
                agent_id=agent_id,
                messages=[HumanMessage(content="Input")],
                step_index=1,
            )
            ctx = collector.before_call(ctx)
            ctx.response = AIMessage(content="Output")
            ctx = collector.after_call(ctx)

        assert len(collector.steps) == 3
        assert collector.get_agent_steps("reasoning")[0].agent_id == "reasoning"

    def test_agentstress_telemetry_collector_fault_tracking(self):
        collector = TelemetryCollector()
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        ctx.fault_applied = "context_truncation"

        ctx = collector.before_call(ctx)
        ctx.response = AIMessage(content="Output")
        ctx = collector.after_call(ctx)

        assert collector.get_fault_count() == 1
        assert collector.steps[0].fault_applied == "context_truncation"

    def test_agentstress_telemetry_collector_summary(self):
        collector = TelemetryCollector()

        for i in range(3):
            ctx = InterceptionContext(
                agent_id=f"agent-{i}",
                messages=[HumanMessage(content="Input")],
                step_index=i,
            )
            ctx = collector.before_call(ctx)
            ctx.response = AIMessage(content="Output")
            ctx = collector.after_call(ctx)

        summary = collector.summary()
        assert summary["total_steps"] == 3
        assert summary["total_tokens"] > 0
        assert len(summary["agents_observed"]) == 3

    def test_agentstress_telemetry_collector_reset(self):
        collector = TelemetryCollector()
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        ctx = collector.before_call(ctx)
        ctx.response = AIMessage(content="Output")
        ctx = collector.after_call(ctx)

        assert len(collector.steps) == 1
        collector.reset()
        assert len(collector.steps) == 0

    def test_agentstress_telemetry_collector_name(self):
        collector = TelemetryCollector()
        assert collector.name == "telemetry-collector"
