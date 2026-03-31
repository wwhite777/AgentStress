"""Abstract FrameworkAdapter: interface for wrapping multi-agent frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from proxy.agentstress_proxy_intercept import InterceptionPipeline
from topology.agentstress_topology_define import TopologySpec


class FrameworkAdapter(ABC):
    """Abstract base for framework-specific adapters.

    An adapter takes an existing multi-agent application (e.g., a LangGraph StateGraph)
    and instruments it with ProxiedChatModels routed through an InterceptionPipeline.
    """

    def __init__(self, topology: TopologySpec, pipeline: InterceptionPipeline) -> None:
        self.topology = topology
        self.pipeline = pipeline

    @abstractmethod
    def wrap(self, app: Any) -> Any:
        """Wrap the framework app with proxied LLMs.

        Returns the instrumented app ready for execution.
        """
        ...

    @abstractmethod
    def extract_agent_ids(self, app: Any) -> list[str]:
        """Extract agent/node IDs from the framework app."""
        ...

    @abstractmethod
    async def run(self, app: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped app with given inputs and return outputs."""
        ...

    @abstractmethod
    async def run_baseline(self, app: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the app WITHOUT fault injection (clean pipeline) for comparison."""
        ...
