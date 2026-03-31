"""Topology definitions: AgentNode, AgentEdge, TopologySpec (Pydantic models)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    CRITIC = "critic"
    ROUTER = "router"
    AGGREGATOR = "aggregator"


class EdgeType(str, Enum):
    DIRECTED = "directed"
    BIDIRECTIONAL = "bidirectional"
    CONDITIONAL = "conditional"


class AgentNode(BaseModel):
    """A single agent in the topology."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable name")
    role: AgentRole = Field(default=AgentRole.WORKER)
    model: str = Field(default="default", description="LLM model identifier")
    description: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_supervisor(self) -> bool:
        return self.role == AgentRole.SUPERVISOR


class AgentEdge(BaseModel):
    """A communication link between two agents."""

    source: str = Field(..., description="Source agent ID")
    target: str = Field(..., description="Target agent ID")
    edge_type: EdgeType = Field(default=EdgeType.DIRECTED)
    label: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopologySpec(BaseModel):
    """Complete topology specification for a multi-agent system."""

    name: str = Field(..., description="Topology name")
    version: str = Field(default="1.0")
    description: str = Field(default="")
    agents: list[AgentNode] = Field(default_factory=list)
    edges: list[AgentEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_agent(self, agent_id: str) -> AgentNode | None:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def get_neighbors(self, agent_id: str) -> list[str]:
        """Get all agents reachable from the given agent."""
        neighbors = []
        for edge in self.edges:
            if edge.source == agent_id:
                neighbors.append(edge.target)
            elif edge.edge_type == EdgeType.BIDIRECTIONAL and edge.target == agent_id:
                neighbors.append(edge.source)
        return neighbors

    def get_upstream(self, agent_id: str) -> list[str]:
        """Get all agents that send messages to the given agent."""
        upstream = []
        for edge in self.edges:
            if edge.target == agent_id:
                upstream.append(edge.source)
            elif edge.edge_type == EdgeType.BIDIRECTIONAL and edge.source == agent_id:
                upstream.append(edge.target)
        return upstream

    def validate_edges(self) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        agent_ids = set(self.get_agent_ids())
        errors = []
        for edge in self.edges:
            if edge.source not in agent_ids:
                errors.append(f"Edge source '{edge.source}' not found in agents")
            if edge.target not in agent_ids:
                errors.append(f"Edge target '{edge.target}' not found in agents")
            if edge.source == edge.target:
                errors.append(f"Self-loop detected on agent '{edge.source}'")
        return errors
