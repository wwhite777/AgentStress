"""Tests for fault injectors (base + context corruption)."""

from __future__ import annotations

import random

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from faults.agentstress_fault_base import FaultConfig, FaultType
from faults.agentstress_fault_context import (
    ContextNoiseFault,
    ContextTruncationFault,
    RAGFailureFault,
    create_context_fault,
)
from proxy.agentstress_proxy_intercept import InterceptionContext


class TestAgentStressFaultConfig:
    """Tests for FaultConfig targeting logic."""

    def test_agentstress_fault_config_target_all(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION)
        assert config.should_target("any_agent")

    def test_agentstress_fault_config_target_specific(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            target_agents=["reasoning"],
        )
        assert config.should_target("reasoning")
        assert not config.should_target("triage")

    def test_agentstress_fault_config_disabled(self):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            enabled=False,
        )
        assert not config.should_target("any_agent")


class TestAgentStressContextTruncation:
    """Tests for ContextTruncationFault."""

    def test_agentstress_fault_truncation_basic(self, agentstress_sample_context):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            params={"keep_ratio": 0.5, "keep_system": True},
        )
        fault = ContextTruncationFault(config)
        ctx = fault.apply_fault(agentstress_sample_context)

        system_count = sum(1 for m in ctx.messages if isinstance(m, SystemMessage))
        assert system_count == 1  # system message preserved
        assert len(ctx.messages) < 4  # messages were truncated

    def test_agentstress_fault_truncation_preserves_system(self, agentstress_sample_messages):
        ctx = InterceptionContext(agent_id="test", messages=agentstress_sample_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            params={"keep_ratio": 0.1, "keep_system": True},
        )
        fault = ContextTruncationFault(config)
        result = fault.apply_fault(ctx)

        assert any(isinstance(m, SystemMessage) for m in result.messages)

    def test_agentstress_fault_truncation_single_message(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Single")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            params={"keep_ratio": 0.1},
        )
        fault = ContextTruncationFault(config)
        result = fault.apply_fault(ctx)
        assert len(result.messages) == 1  # should not drop the only message


class TestAgentStressContextNoise:
    """Tests for ContextNoiseFault."""

    def test_agentstress_fault_noise_corrupts_human_messages(self):
        random.seed(42)
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="A" * 100),
        ]
        ctx = InterceptionContext(agent_id="test", messages=messages, step_index=1)

        config = FaultConfig(
            fault_type=FaultType.CONTEXT_NOISE,
            probability=1.0,
            params={"noise_ratio": 0.5, "target_role": "human"},
        )
        fault = ContextNoiseFault(config)
        result = fault.apply_fault(ctx)

        assert result.messages[0].content == "System prompt"  # system untouched
        assert result.messages[1].content != "A" * 100  # human corrupted
        assert result.metadata["noise_corrupted_messages"] == 1

    def test_agentstress_fault_noise_target_all(self):
        random.seed(42)
        messages = [
            HumanMessage(content="AAAA"),
            AIMessage(content="BBBB"),
        ]
        ctx = InterceptionContext(agent_id="test", messages=messages, step_index=1)

        config = FaultConfig(
            fault_type=FaultType.CONTEXT_NOISE,
            probability=1.0,
            params={"noise_ratio": 1.0, "target_role": "all"},
        )
        fault = ContextNoiseFault(config)
        result = fault.apply_fault(ctx)
        assert result.metadata["noise_corrupted_messages"] == 2


class TestAgentStressRAGFailure:
    """Tests for RAGFailureFault."""

    def test_agentstress_fault_rag_empty_mode(self, agentstress_sample_messages):
        ctx = InterceptionContext(agent_id="test", messages=agentstress_sample_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.RAG_FAILURE,
            probability=1.0,
            params={"failure_mode": "empty", "rag_marker": "[Retrieved Context]"},
        )
        fault = RAGFailureFault(config)
        result = fault.apply_fault(ctx)

        rag_msg = [m for m in result.messages if "[Retrieved Context]" in m.content]
        assert len(rag_msg) == 1
        assert "No results found" in rag_msg[0].content
        assert result.metadata["rag_failures_injected"] == 1

    def test_agentstress_fault_rag_irrelevant_mode(self, agentstress_sample_messages):
        random.seed(42)
        ctx = InterceptionContext(agent_id="test", messages=agentstress_sample_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.RAG_FAILURE,
            probability=1.0,
            params={"failure_mode": "irrelevant", "rag_marker": "[Retrieved Context]"},
        )
        fault = RAGFailureFault(config)
        result = fault.apply_fault(ctx)

        rag_msg = [m for m in result.messages if "[Retrieved Context]" in m.content]
        assert "ACS guidelines" not in rag_msg[0].content  # original content replaced

    def test_agentstress_fault_rag_no_marker(self):
        messages = [HumanMessage(content="No RAG content here")]
        ctx = InterceptionContext(agent_id="test", messages=messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.RAG_FAILURE,
            probability=1.0,
            params={"failure_mode": "empty"},
        )
        fault = RAGFailureFault(config)
        result = fault.apply_fault(ctx)
        assert result.metadata["rag_failures_injected"] == 0


class TestAgentStressFaultFactory:
    """Tests for the create_context_fault factory."""

    def test_agentstress_fault_factory_truncation(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION)
        fault = create_context_fault(config)
        assert isinstance(fault, ContextTruncationFault)

    def test_agentstress_fault_factory_noise(self):
        config = FaultConfig(fault_type=FaultType.CONTEXT_NOISE)
        fault = create_context_fault(config)
        assert isinstance(fault, ContextNoiseFault)

    def test_agentstress_fault_factory_rag(self):
        config = FaultConfig(fault_type=FaultType.RAG_FAILURE)
        fault = create_context_fault(config)
        assert isinstance(fault, RAGFailureFault)

    def test_agentstress_fault_factory_unknown(self):
        config = FaultConfig(fault_type=FaultType.BYZANTINE)
        with pytest.raises(ValueError, match="Unknown context fault"):
            create_context_fault(config)


class TestAgentStressFaultProbability:
    """Tests for probability-based fault triggering."""

    def test_agentstress_fault_always_triggers(self, agentstress_sample_context):
        random.seed(42)
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=1.0,
            params={"keep_ratio": 0.5},
        )
        fault = ContextTruncationFault(config)

        triggered = 0
        for _ in range(10):
            ctx = InterceptionContext(
                agent_id="reasoning",
                messages=list(agentstress_sample_context.messages),
                step_index=1,
            )
            result = fault.before_call(ctx)
            if result.fault_applied:
                triggered += 1
        assert triggered == 10

    def test_agentstress_fault_never_triggers(self, agentstress_sample_context):
        config = FaultConfig(
            fault_type=FaultType.CONTEXT_TRUNCATION,
            probability=0.0,
            params={"keep_ratio": 0.5},
        )
        fault = ContextTruncationFault(config)
        ctx = fault.before_call(agentstress_sample_context)
        assert ctx.fault_applied is None
