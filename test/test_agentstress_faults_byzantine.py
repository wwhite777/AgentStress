"""Tests for byzantine fault injectors (rogue agent, hallucination)."""

from __future__ import annotations

import random

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from faults.agentstress_fault_base import FaultConfig, FaultType
from faults.agentstress_fault_byzantine import (
    ByzantineFault,
    HallucinationFault,
    create_byzantine_fault,
)
from proxy.agentstress_proxy_intercept import InterceptionContext


class TestAgentStressByzantineFault:
    """Tests for ByzantineFault."""

    def test_agentstress_fault_byzantine_full_replace(self):
        random.seed(42)
        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Analyze patient data")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.BYZANTINE,
            probability=1.0,
            params={"replace_mode": "full"},
        )
        fault = ByzantineFault(config)

        ctx = fault.before_call(ctx)
        assert ctx.fault_applied == "byzantine"
        assert "byzantine_rogue_text" in ctx.metadata

        ctx.response = AIMessage(content="Original analysis result")
        ctx = fault.after_call(ctx)

        assert ctx.response.content != "Original analysis result"
        assert len(ctx.response.content) > 0

    def test_agentstress_fault_byzantine_prepend(self):
        random.seed(42)
        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.BYZANTINE,
            probability=1.0,
            params={"replace_mode": "prepend"},
        )
        fault = ByzantineFault(config)

        ctx = fault.before_call(ctx)
        ctx.response = AIMessage(content="Original text")
        ctx = fault.after_call(ctx)

        assert "Original text" in ctx.response.content
        assert len(ctx.response.content) > len("Original text")

    def test_agentstress_fault_byzantine_custom_responses(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.BYZANTINE,
            probability=1.0,
            params={"rogue_responses": ["CUSTOM ROGUE OUTPUT"], "replace_mode": "full"},
        )
        fault = ByzantineFault(config)

        ctx = fault.before_call(ctx)
        ctx.response = AIMessage(content="Original")
        ctx = fault.after_call(ctx)

        assert ctx.response.content == "CUSTOM ROGUE OUTPUT"

    def test_agentstress_fault_byzantine_no_trigger_on_other_fault(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        ctx.fault_applied = "context_truncation"  # different fault
        config = FaultConfig(fault_type=FaultType.BYZANTINE, probability=1.0)
        fault = ByzantineFault(config)

        ctx.response = AIMessage(content="Original")
        ctx = fault.after_call(ctx)
        assert ctx.response.content == "Original"


class TestAgentStressHallucinationFault:
    """Tests for HallucinationFault."""

    def test_agentstress_fault_hallucination_append(self):
        random.seed(42)
        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.HALLUCINATION,
            probability=1.0,
            params={"num_hallucinations": 2, "injection_point": "append"},
        )
        fault = HallucinationFault(config)

        ctx = fault.before_call(ctx)
        assert len(ctx.metadata["hallucinations_prepared"]) == 2

        ctx.response = AIMessage(content="Real analysis.")
        ctx = fault.after_call(ctx)

        assert "Real analysis." in ctx.response.content
        assert ctx.metadata["hallucinations_injected"] == 2

    def test_agentstress_fault_hallucination_replace(self):
        random.seed(42)
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.HALLUCINATION,
            probability=1.0,
            params={"num_hallucinations": 1, "injection_point": "replace"},
        )
        fault = HallucinationFault(config)

        ctx = fault.before_call(ctx)
        ctx.response = AIMessage(content="Original")
        ctx = fault.after_call(ctx)

        assert "Original" not in ctx.response.content

    def test_agentstress_fault_hallucination_contains_fake_citations(self):
        random.seed(42)
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.HALLUCINATION,
            probability=1.0,
            params={"num_hallucinations": 3, "injection_point": "replace"},
        )
        fault = HallucinationFault(config)

        ctx = fault.before_call(ctx)
        ctx.response = AIMessage(content="Original")
        ctx = fault.after_call(ctx)

        content = ctx.response.content
        has_citation = any(
            marker in content
            for marker in ["doi:", "FDA", "meta-analysis", "WHO", "study", "trials"]
        )
        assert has_citation


class TestAgentStressByzantineFaultFactory:
    """Tests for create_byzantine_fault factory."""

    def test_agentstress_fault_factory_byzantine(self):
        config = FaultConfig(fault_type=FaultType.BYZANTINE)
        assert isinstance(create_byzantine_fault(config), ByzantineFault)

    def test_agentstress_fault_factory_hallucination(self):
        config = FaultConfig(fault_type=FaultType.HALLUCINATION)
        assert isinstance(create_byzantine_fault(config), HallucinationFault)

    def test_agentstress_fault_factory_unknown(self):
        config = FaultConfig(fault_type=FaultType.DEADLOCK)
        with pytest.raises(ValueError, match="Unknown byzantine fault"):
            create_byzantine_fault(config)
