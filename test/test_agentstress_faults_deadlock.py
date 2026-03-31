"""Tests for deadlock/livelock fault injectors."""

from __future__ import annotations

import random

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from faults.agentstress_fault_base import FaultConfig, FaultType
from faults.agentstress_fault_deadlock import (
    DeadlockFault,
    TokenThrashFault,
    create_deadlock_fault,
)
from proxy.agentstress_proxy_intercept import InterceptionContext


class TestAgentStressDeadlockFault:
    """Tests for DeadlockFault."""

    def test_agentstress_fault_deadlock_delegation(self):
        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Analyze data")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.DEADLOCK,
            probability=1.0,
            params={"delegate_to": "literature", "max_loops": 3},
        )
        fault = DeadlockFault(config)

        ctx = fault.before_call(ctx)
        assert ctx.metadata["deadlock_loop_count"] == 1

        ctx.response = AIMessage(content="Original response")
        ctx = fault.after_call(ctx)

        assert "literature" in ctx.response.content
        assert "information" in ctx.response.content.lower() or "re-analyze" in ctx.response.content.lower()

    def test_agentstress_fault_deadlock_self_terminates(self):
        config = FaultConfig(
            fault_type=FaultType.DEADLOCK,
            probability=1.0,
            params={"max_loops": 2},
        )
        fault = DeadlockFault(config)

        for i in range(3):
            ctx = InterceptionContext(
                agent_id="reasoning",
                messages=[HumanMessage(content="Input")],
                step_index=i,
            )
            ctx = fault.before_call(ctx)

        assert ctx.metadata.get("deadlock_terminated")
        assert ctx.metadata["deadlock_loops"] == 2

    def test_agentstress_fault_deadlock_per_agent_tracking(self):
        config = FaultConfig(
            fault_type=FaultType.DEADLOCK,
            probability=1.0,
            params={"max_loops": 5},
        )
        fault = DeadlockFault(config)

        for agent_id in ["agent_a", "agent_b"]:
            ctx = InterceptionContext(
                agent_id=agent_id,
                messages=[HumanMessage(content="Input")],
                step_index=1,
            )
            ctx = fault.before_call(ctx)
            assert ctx.metadata["deadlock_loop_count"] == 1

    def test_agentstress_fault_deadlock_reset(self):
        config = FaultConfig(
            fault_type=FaultType.DEADLOCK,
            probability=1.0,
            params={"max_loops": 2},
        )
        fault = DeadlockFault(config)

        ctx = InterceptionContext(agent_id="test", messages=[HumanMessage(content="Input")], step_index=1)
        fault.before_call(ctx)
        fault.before_call(ctx)

        fault.reset()

        ctx2 = InterceptionContext(agent_id="test", messages=[HumanMessage(content="Input")], step_index=1)
        result = fault.before_call(ctx2)
        assert result.metadata.get("deadlock_loop_count") == 1  # reset worked


class TestAgentStressTokenThrashFault:
    """Tests for TokenThrashFault."""

    def test_agentstress_fault_token_thrash_inflates_input(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Short input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.TOKEN_THRASH,
            probability=1.0,
            params={"inflation_factor": 3, "target_input": True, "target_output": False},
        )
        fault = TokenThrashFault(config)
        result = fault.apply_fault(ctx)

        inflated_content = result.messages[0].content
        assert len(inflated_content) > len("Short input") * 2
        assert result.metadata["token_thrash_input_inflated"]

    def test_agentstress_fault_token_thrash_inflates_output(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        ctx.fault_applied = FaultType.TOKEN_THRASH.value
        config = FaultConfig(
            fault_type=FaultType.TOKEN_THRASH,
            probability=1.0,
            params={"inflation_factor": 3, "target_input": False, "target_output": True},
        )
        fault = TokenThrashFault(config)

        ctx.response = AIMessage(content="Short response")
        result = fault.after_call(ctx)

        assert len(result.response.content) > len("Short response") * 2

    def test_agentstress_fault_token_thrash_custom_padding(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.TOKEN_THRASH,
            probability=1.0,
            params={"inflation_factor": 2, "padding_text": "PADDING"},
        )
        fault = TokenThrashFault(config)
        result = fault.apply_fault(ctx)

        assert "PADDING" in result.messages[0].content

    def test_agentstress_fault_token_thrash_skips_system(self):
        from langchain_core.messages import SystemMessage

        ctx = InterceptionContext(
            agent_id="test",
            messages=[SystemMessage(content="System"), HumanMessage(content="Human")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.TOKEN_THRASH,
            probability=1.0,
            params={"inflation_factor": 3},
        )
        fault = TokenThrashFault(config)
        result = fault.apply_fault(ctx)

        assert result.messages[0].content == "System"  # system unchanged
        assert len(result.messages[1].content) > len("Human")  # human inflated


class TestAgentStressDeadlockFaultFactory:
    """Tests for create_deadlock_fault factory."""

    def test_agentstress_fault_factory_deadlock(self):
        config = FaultConfig(fault_type=FaultType.DEADLOCK)
        assert isinstance(create_deadlock_fault(config), DeadlockFault)

    def test_agentstress_fault_factory_token_thrash(self):
        config = FaultConfig(fault_type=FaultType.TOKEN_THRASH)
        assert isinstance(create_deadlock_fault(config), TokenThrashFault)

    def test_agentstress_fault_factory_unknown(self):
        config = FaultConfig(fault_type=FaultType.BYZANTINE)
        with pytest.raises(ValueError, match="Unknown deadlock fault"):
            create_deadlock_fault(config)
