"""Tests for fault scheduling (continuous, burst, progressive, once)."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from faults.agentstress_fault_base import FaultConfig, FaultSchedule, FaultType
from faults.agentstress_fault_context import ContextTruncationFault
from faults.agentstress_fault_schedule import ScheduledFaultWrapper, wrap_with_schedule
from proxy.agentstress_proxy_intercept import InterceptionContext


def _make_ctx(agent_id: str = "test") -> InterceptionContext:
    return InterceptionContext(
        agent_id=agent_id,
        messages=[HumanMessage(content="Message 1"), HumanMessage(content="Message 2")],
        step_index=1,
    )


class TestAgentStressScheduleContinuous:
    """Tests for continuous schedule (always active)."""

    def test_agentstress_schedule_continuous_always_active(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            schedule=FaultSchedule.CONTINUOUS,
            params={"keep_ratio": 0.5},
        )
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        triggered = 0
        for _ in range(5):
            ctx = _make_ctx()
            result = scheduled.before_call(ctx)
            if result.fault_applied:
                triggered += 1

        assert triggered == 5


class TestAgentStressScheduleOnce:
    """Tests for once schedule (trigger exactly once)."""

    def test_agentstress_schedule_once_triggers_once(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            schedule=FaultSchedule.ONCE,
            params={"keep_ratio": 0.5},
        )
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        triggered = 0
        for _ in range(5):
            ctx = _make_ctx()
            result = scheduled.before_call(ctx)
            if result.fault_applied:
                triggered += 1

        assert triggered == 1

    def test_agentstress_schedule_once_reset(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            schedule=FaultSchedule.ONCE,
            params={"keep_ratio": 0.5},
        )
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        ctx = _make_ctx()
        scheduled.before_call(ctx)
        assert scheduled.triggered

        scheduled.reset()
        assert not scheduled.triggered


class TestAgentStressScheduleBurst:
    """Tests for burst schedule (on for N, off for M, repeat)."""

    def test_agentstress_schedule_burst_pattern(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            schedule=FaultSchedule.BURST,
            params={"keep_ratio": 0.5, "burst_on_steps": 2, "burst_off_steps": 3},
        )
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        pattern = []
        for _ in range(10):
            ctx = _make_ctx()
            result = scheduled.before_call(ctx)
            pattern.append(result.fault_applied is not None)

        # cycle=5: steps 1,2 on; steps 3,4,5 off; steps 6,7 on; etc.
        # but step_count starts at 1 after increment, so position = (step-1) % 5
        # step 1: pos 0 (on), step 2: pos 1 (on), step 3: pos 2 (off), ...
        assert pattern[0] is True   # step 1: on
        assert pattern[1] is True   # step 2: on
        assert pattern[2] is False  # step 3: off
        assert pattern[3] is False  # step 4: off
        assert pattern[4] is False  # step 5: off
        assert pattern[5] is True   # step 6: on (cycle repeats)


class TestAgentStressScheduleProgressive:
    """Tests for progressive schedule (probability ramps up)."""

    def test_agentstress_schedule_progressive_ramps(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,  # base probability (overridden by progressive)
            schedule=FaultSchedule.PROGRESSIVE,
            params={
                "keep_ratio": 0.5,
                "progressive_start": 0.0,
                "progressive_end": 1.0,
                "progressive_ramp_steps": 10,
            },
        )
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        # At step 1, effective prob = 0.1, at step 10, effective prob = 1.0
        ctx = _make_ctx()
        scheduled.before_call(ctx)  # step 1
        assert "schedule_effective_probability" in ctx.metadata
        prob_1 = ctx.metadata["schedule_effective_probability"]

        # Run to step 10
        for _ in range(9):
            ctx = _make_ctx()
            scheduled.before_call(ctx)

        prob_10 = ctx.metadata["schedule_effective_probability"]
        assert prob_10 > prob_1
        assert prob_10 == pytest.approx(1.0, abs=0.01)


class TestAgentStressScheduledWrapper:
    """General tests for ScheduledFaultWrapper."""

    def test_agentstress_schedule_name(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION, params={"keep_ratio": 0.5})
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)
        assert "scheduled" in scheduled.name
        assert "context_truncation" in scheduled.name

    def test_agentstress_schedule_step_count(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION, params={"keep_ratio": 0.5})
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        for _ in range(3):
            ctx = _make_ctx()
            scheduled.before_call(ctx)

        assert scheduled.step_count == 3

    def test_agentstress_schedule_after_call_passthrough(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION, params={"keep_ratio": 0.5})
        fault = ContextTruncationFault(config)
        scheduled = wrap_with_schedule(fault)

        ctx = _make_ctx()
        result = scheduled.after_call(ctx)
        assert result is ctx
