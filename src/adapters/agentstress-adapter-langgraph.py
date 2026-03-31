"""LangGraph-specific adapter: wraps StateGraph nodes with ProxiedChatModels."""

from __future__ import annotations

import copy
import importlib
import inspect
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from adapters.agentstress_adapter_base import FrameworkAdapter
from proxy.agentstress_proxy_intercept import InterceptionContext, InterceptionPipeline, Interceptor
from proxy.agentstress_proxy_llm import ProxiedChatModel
from topology.agentstress_topology_define import TopologySpec


class NoopInterceptor:
    """Passthrough interceptor used for baseline runs."""

    @property
    def name(self) -> str:
        return "noop"

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        return ctx


class LangGraphAdapter(FrameworkAdapter):
    """Adapter for LangGraph StateGraph applications.

    Discovers BaseChatModel instances used by graph nodes and replaces them
    with ProxiedChatModel wrappers routed through the InterceptionPipeline.

    Works with both compiled and uncompiled StateGraph objects.
    """

    def __init__(self, topology: TopologySpec, pipeline: InterceptionPipeline) -> None:
        super().__init__(topology, pipeline)
        self._original_models: dict[str, BaseChatModel] = {}
        self._proxied_models: dict[str, ProxiedChatModel] = {}

    def extract_agent_ids(self, app: Any) -> list[str]:
        """Extract node names from a LangGraph StateGraph or CompiledGraph."""
        if hasattr(app, "nodes"):
            nodes = app.nodes
            if isinstance(nodes, dict):
                return [k for k in nodes.keys() if k not in ("__start__", "__end__")]
            return list(nodes)

        if hasattr(app, "builder") and hasattr(app.builder, "nodes"):
            return [k for k in app.builder.nodes.keys() if k not in ("__start__", "__end__")]

        return self.topology.get_agent_ids()

    def wrap(self, app: Any) -> Any:
        """Wrap LangGraph app nodes with ProxiedChatModels.

        Scans each node's callable for BaseChatModel attributes and replaces
        them with proxied versions. Returns the original app (mutated in-place).
        """
        agent_ids = self.extract_agent_ids(app)

        nodes = {}
        if hasattr(app, "nodes") and isinstance(app.nodes, dict):
            nodes = app.nodes
        elif hasattr(app, "builder") and hasattr(app.builder, "nodes"):
            nodes = app.builder.nodes

        for node_name in agent_ids:
            if node_name not in nodes:
                continue

            node_obj = nodes[node_name]
            self._wrap_node_models(node_name, node_obj)

        return app

    def _wrap_node_models(self, agent_id: str, node_obj: Any) -> None:
        """Find and wrap BaseChatModel instances in a node object."""
        targets = [node_obj]
        if hasattr(node_obj, "func"):
            targets.append(node_obj.func)
        if hasattr(node_obj, "__self__"):
            targets.append(node_obj.__self__)

        for target in targets:
            for attr_name in dir(target):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(target, attr_name)
                except (AttributeError, Exception):
                    continue

                if isinstance(attr, BaseChatModel) and not isinstance(attr, ProxiedChatModel):
                    proxied = ProxiedChatModel(
                        wrapped=attr,
                        pipeline=self.pipeline,
                        agent_id=agent_id,
                    )
                    self._original_models[agent_id] = attr
                    self._proxied_models[agent_id] = proxied
                    try:
                        setattr(target, attr_name, proxied)
                    except (AttributeError, TypeError):
                        pass

    async def run(self, app: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped app with fault-injected pipeline."""
        if hasattr(app, "ainvoke"):
            return await app.ainvoke(inputs)
        if hasattr(app, "invoke"):
            return app.invoke(inputs)
        raise TypeError(f"App type {type(app)} does not support invoke/ainvoke")

    async def run_baseline(self, app: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the app with a clean (no-fault) pipeline for baseline comparison."""
        original_interceptors = self.pipeline.interceptors
        self.pipeline.clear()
        self.pipeline.add(NoopInterceptor())

        try:
            result = await self.run(app, inputs)
        finally:
            self.pipeline.clear()
            for interceptor in original_interceptors:
                self.pipeline.add(interceptor)

        return result

    def get_proxied_models(self) -> dict[str, ProxiedChatModel]:
        return dict(self._proxied_models)

    def get_original_models(self) -> dict[str, BaseChatModel]:
        return dict(self._original_models)
