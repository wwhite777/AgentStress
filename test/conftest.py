"""Shared test fixtures: sample topologies, mock LLMs, stub graphs."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import sys
from pathlib import Path

# Add src/ to path once, then import all packages to trigger __init__.py loading
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Import all packages to trigger their __init__.py importlib loaders.
# This registers all hyphenated modules in sys.modules so cross-package
# imports (e.g., from topology.agentstress_topology_define import ...) work.
import topology  # noqa: F401
import proxy  # noqa: F401
import faults  # noqa: F401
import telemetry  # noqa: F401
import adapters  # noqa: F401
import eval  # noqa: F401
import runner  # noqa: F401
import replay  # noqa: F401

from faults.agentstress_fault_base import FaultConfig, FaultType
from proxy.agentstress_proxy_intercept import InterceptionContext, InterceptionPipeline
from topology.agentstress_topology_define import (
    AgentEdge,
    AgentNode,
    AgentRole,
    EdgeType,
    TopologySpec,
)


class StubChatModel(BaseChatModel):
    """Deterministic chat model for testing — always returns a fixed response."""

    response_text: str = "Stub response from agent."

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response_text))]
        )


@pytest.fixture
def agentstress_sample_topology() -> TopologySpec:
    """3-agent linear topology: triage → reasoning → report."""
    return TopologySpec(
        name="test-linear",
        agents=[
            AgentNode(id="triage", name="Triage", role=AgentRole.ROUTER),
            AgentNode(id="reasoning", name="Reasoning", role=AgentRole.WORKER),
            AgentNode(id="report", name="Report", role=AgentRole.AGGREGATOR),
        ],
        edges=[
            AgentEdge(source="triage", target="reasoning"),
            AgentEdge(source="reasoning", target="report"),
        ],
    )


@pytest.fixture
def agentstress_medagent_topology() -> TopologySpec:
    """4-agent MedAgent topology matching the example config."""
    return TopologySpec(
        name="medagent-clinical",
        agents=[
            AgentNode(id="triage", name="Triage Agent", role=AgentRole.ROUTER),
            AgentNode(id="literature", name="Literature Agent", role=AgentRole.WORKER),
            AgentNode(id="reasoning", name="Reasoning Agent", role=AgentRole.WORKER),
            AgentNode(id="report", name="Report Agent", role=AgentRole.AGGREGATOR),
        ],
        edges=[
            AgentEdge(source="triage", target="literature"),
            AgentEdge(source="triage", target="reasoning"),
            AgentEdge(source="literature", target="reasoning"),
            AgentEdge(source="reasoning", target="report"),
        ],
    )


@pytest.fixture
def agentstress_stub_llm() -> StubChatModel:
    return StubChatModel(response_text="Test LLM response.")


@pytest.fixture
def agentstress_pipeline() -> InterceptionPipeline:
    return InterceptionPipeline()


@pytest.fixture
def agentstress_sample_messages() -> list[BaseMessage]:
    return [
        SystemMessage(content="You are a clinical reasoning agent."),
        HumanMessage(content="Patient presents with chest pain and shortness of breath."),
        AIMessage(content="Initial assessment suggests cardiac evaluation needed."),
        HumanMessage(content="[Retrieved Context]\nACS guidelines recommend troponin testing."),
    ]


@pytest.fixture
def agentstress_sample_context(agentstress_sample_messages) -> InterceptionContext:
    return InterceptionContext(
        agent_id="reasoning",
        messages=agentstress_sample_messages,
        step_index=1,
    )
