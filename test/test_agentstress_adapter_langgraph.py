"""Tests for LangGraph adapter and topology visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from adapters.agentstress_adapter_langgraph import LangGraphAdapter, NoopInterceptor
from proxy.agentstress_proxy_intercept import InterceptionContext, InterceptionPipeline
from proxy.agentstress_proxy_llm import ProxiedChatModel
from topology.agentstress_topology_define import (
    AgentEdge,
    AgentNode,
    AgentRole,
    TopologySpec,
)
from topology.agentstress_topology_visualize import (
    to_graphviz,
    to_mermaid,
    save_mermaid,
    save_graphviz,
)


class _StubModel(BaseChatModel):
    response: str = "stub"

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


class _FakeNode:
    def __init__(self, model: BaseChatModel):
        self.llm = model


class _FakeGraph:
    def __init__(self, node_names: list[str]):
        self.nodes = {}
        for name in node_names:
            self.nodes[name] = _FakeNode(_StubModel(response=f"Response from {name}"))

    async def ainvoke(self, inputs: dict) -> dict:
        return {"output": "Graph output", **inputs}

    def invoke(self, inputs: dict) -> dict:
        return {"output": "Graph output", **inputs}


# --- LangGraphAdapter ---

class TestAgentStressLangGraphAdapter:

    @pytest.fixture
    def agentstress_adapter_setup(self):
        topology = TopologySpec(
            name="test",
            agents=[
                AgentNode(id="a", name="Agent A", role=AgentRole.WORKER),
                AgentNode(id="b", name="Agent B", role=AgentRole.WORKER),
            ],
            edges=[AgentEdge(source="a", target="b")],
        )
        pipeline = InterceptionPipeline()
        adapter = LangGraphAdapter(topology, pipeline)
        graph = _FakeGraph(["a", "b"])
        return adapter, graph, pipeline

    def test_agentstress_adapter_extract_agent_ids(self, agentstress_adapter_setup):
        adapter, graph, _ = agentstress_adapter_setup
        ids = adapter.extract_agent_ids(graph)
        assert "a" in ids
        assert "b" in ids

    def test_agentstress_adapter_wrap_replaces_models(self, agentstress_adapter_setup):
        adapter, graph, _ = agentstress_adapter_setup
        adapter.wrap(graph)

        proxied = adapter.get_proxied_models()
        assert len(proxied) >= 1  # at least one model wrapped

    def test_agentstress_adapter_wrap_preserves_originals(self, agentstress_adapter_setup):
        adapter, graph, _ = agentstress_adapter_setup
        adapter.wrap(graph)

        originals = adapter.get_original_models()
        for agent_id, model in originals.items():
            assert isinstance(model, _StubModel)

    @pytest.mark.asyncio
    async def test_agentstress_adapter_run(self, agentstress_adapter_setup):
        adapter, graph, _ = agentstress_adapter_setup
        result = await adapter.run(graph, {"input": "test"})
        assert "output" in result

    @pytest.mark.asyncio
    async def test_agentstress_adapter_run_baseline(self, agentstress_adapter_setup):
        adapter, graph, pipeline = agentstress_adapter_setup

        # Add a counting interceptor to verify it gets cleared during baseline
        class Counter:
            count = 0
            @property
            def name(self):
                return "counter"
            def before_call(self, ctx):
                self.count += 1
                return ctx
            def after_call(self, ctx):
                return ctx

        counter = Counter()
        pipeline.add(counter)

        result = await adapter.run_baseline(graph, {"input": "test"})
        assert "output" in result

        # Pipeline should be restored after baseline
        assert any(i.name == "counter" for i in pipeline.interceptors)


class TestAgentStressNoopInterceptor:

    def test_agentstress_noop_passthrough(self):
        noop = NoopInterceptor()
        ctx = InterceptionContext(
            agent_id="test",
            messages=[HumanMessage(content="Hello")],
            step_index=1,
        )
        result = noop.before_call(ctx)
        assert result is ctx
        result = noop.after_call(ctx)
        assert result is ctx
        assert noop.name == "noop"


# --- Topology Visualization ---

class TestAgentStressTopologyVisualize:

    @pytest.fixture
    def agentstress_viz_topology(self):
        return TopologySpec(
            name="viz-test",
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

    def test_agentstress_viz_mermaid(self, agentstress_viz_topology):
        result = to_mermaid(agentstress_viz_topology)
        assert "graph LR" in result
        assert "triage" in result
        assert "reasoning" in result
        assert "report" in result
        assert "-->" in result

    def test_agentstress_viz_mermaid_td(self, agentstress_viz_topology):
        result = to_mermaid(agentstress_viz_topology, direction="TD")
        assert "graph TD" in result

    def test_agentstress_viz_graphviz(self, agentstress_viz_topology):
        result = to_graphviz(agentstress_viz_topology)
        assert "digraph AgentStress" in result
        assert "triage" in result
        assert "->" in result
        assert "hexagon" in result  # router shape

    def test_agentstress_viz_save_mermaid(self, agentstress_viz_topology, tmp_path):
        path = save_mermaid(agentstress_viz_topology, tmp_path / "topo.md")
        assert path.exists()
        content = path.read_text()
        assert "```mermaid" in content

    def test_agentstress_viz_save_graphviz(self, agentstress_viz_topology, tmp_path):
        path = save_graphviz(agentstress_viz_topology, tmp_path / "topo.dot")
        assert path.exists()
        content = path.read_text()
        assert "digraph" in content

    def test_agentstress_viz_role_styles(self, agentstress_viz_topology):
        result = to_mermaid(agentstress_viz_topology)
        assert "style triage" in result
        assert "style reasoning" in result
        assert "style report" in result
