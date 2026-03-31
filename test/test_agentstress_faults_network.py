"""Tests for network fault injectors (message drop, API throttle)."""

from __future__ import annotations

import random
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from faults.agentstress_fault_base import FaultConfig, FaultType
from faults.agentstress_fault_network import (
    APIThrottleFault,
    MessageDropFault,
    create_network_fault,
)
from proxy.agentstress_proxy_intercept import InterceptionContext


@pytest.fixture
def agentstress_network_messages():
    return [
        SystemMessage(content="System prompt"),
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Message 3"),
    ]


class TestAgentStressMessageDrop:
    """Tests for MessageDropFault."""

    def test_agentstress_fault_message_drop_basic(self, agentstress_network_messages):
        random.seed(42)
        ctx = InterceptionContext(agent_id="test", messages=agentstress_network_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.MESSAGE_DROP,
            probability=1.0,
            params={"drop_ratio": 0.5, "min_keep": 1},
        )
        fault = MessageDropFault(config)
        result = fault.apply_fault(ctx)

        system_count = sum(1 for m in result.messages if isinstance(m, SystemMessage))
        assert system_count == 1
        non_system = [m for m in result.messages if not isinstance(m, SystemMessage)]
        assert len(non_system) < 5
        assert result.metadata["messages_dropped"] > 0

    def test_agentstress_fault_message_drop_preserves_min(self, agentstress_network_messages):
        ctx = InterceptionContext(agent_id="test", messages=agentstress_network_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.MESSAGE_DROP,
            probability=1.0,
            params={"drop_ratio": 0.99, "min_keep": 2},
        )
        fault = MessageDropFault(config)
        result = fault.apply_fault(ctx)

        non_system = [m for m in result.messages if not isinstance(m, SystemMessage)]
        assert len(non_system) >= 2

    def test_agentstress_fault_message_drop_newest_first(self, agentstress_network_messages):
        ctx = InterceptionContext(agent_id="test", messages=agentstress_network_messages, step_index=1)
        config = FaultConfig(
            fault_type=FaultType.MESSAGE_DROP,
            probability=1.0,
            params={"drop_ratio": 0.6, "drop_newest": True, "min_keep": 1},
        )
        fault = MessageDropFault(config)
        result = fault.apply_fault(ctx)

        non_system = [m for m in result.messages if not isinstance(m, SystemMessage)]
        assert non_system[0].content == "Message 1"

    def test_agentstress_fault_message_drop_too_few_messages(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[SystemMessage(content="Sys"), HumanMessage(content="Only one")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.MESSAGE_DROP,
            probability=1.0,
            params={"drop_ratio": 0.5, "min_keep": 1},
        )
        fault = MessageDropFault(config)
        result = fault.apply_fault(ctx)
        assert len(result.messages) == 2  # no drop, only 1 non-system


class TestAgentStressAPIThrottle:
    """Tests for APIThrottleFault."""

    def test_agentstress_fault_throttle_latency(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Hello")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.API_THROTTLE,
            probability=1.0,
            params={"mode": "latency", "latency_ms": 50},
        )
        fault = APIThrottleFault(config)
        start = time.time()
        result = fault.apply_fault(ctx)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms >= 40  # allow small tolerance
        assert result.metadata["throttle_delay_ms"] == 50
        assert not result.skipped

    def test_agentstress_fault_throttle_error(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Hello")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.API_THROTTLE,
            probability=1.0,
            params={"mode": "error"},
        )
        fault = APIThrottleFault(config)
        result = fault.apply_fault(ctx)

        assert result.skipped
        assert "Rate limit" in result.response.content
        assert result.metadata["throttle_error"]

    def test_agentstress_fault_throttle_custom_error(self):
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Hello")],
            step_index=1,
        )
        config = FaultConfig(
            fault_type=FaultType.API_THROTTLE,
            probability=1.0,
            params={"mode": "error", "error_message": "Service unavailable (503)"},
        )
        fault = APIThrottleFault(config)
        result = fault.apply_fault(ctx)

        assert result.response.content == "Service unavailable (503)"


class TestAgentStressNetworkFaultFactory:
    """Tests for create_network_fault factory."""

    def test_agentstress_fault_factory_message_drop(self):
        config = FaultConfig(fault_type=FaultType.MESSAGE_DROP)
        assert isinstance(create_network_fault(config), MessageDropFault)

    def test_agentstress_fault_factory_api_throttle(self):
        config = FaultConfig(fault_type=FaultType.API_THROTTLE)
        assert isinstance(create_network_fault(config), APIThrottleFault)

    def test_agentstress_fault_factory_unknown(self):
        config = FaultConfig(fault_type=FaultType.BYZANTINE)
        with pytest.raises(ValueError, match="Unknown network fault"):
            create_network_fault(config)
