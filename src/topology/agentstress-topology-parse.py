"""YAML loader and factory functions for topology specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from topology.agentstress_topology_define import (
    AgentEdge,
    AgentNode,
    AgentRole,
    EdgeType,
    TopologySpec,
)


def load_topology_yaml(path: str | Path) -> TopologySpec:
    """Load a TopologySpec from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Topology file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return parse_topology_dict(raw)


def parse_topology_dict(raw: dict[str, Any]) -> TopologySpec:
    """Parse a raw dictionary into a TopologySpec."""
    topology_data = raw.get("topology", raw)

    agents = []
    for agent_raw in topology_data.get("agents", []):
        role_str = agent_raw.get("role", "worker")
        agents.append(
            AgentNode(
                id=agent_raw["id"],
                name=agent_raw.get("name", agent_raw["id"]),
                role=AgentRole(role_str),
                model=agent_raw.get("model", "default"),
                description=agent_raw.get("description", ""),
                metadata=agent_raw.get("metadata", {}),
            )
        )

    edges = []
    for edge_raw in topology_data.get("edges", []):
        edge_type_str = edge_raw.get("edge_type", "directed")
        edges.append(
            AgentEdge(
                source=edge_raw["source"],
                target=edge_raw["target"],
                edge_type=EdgeType(edge_type_str),
                label=edge_raw.get("label", ""),
                metadata=edge_raw.get("metadata", {}),
            )
        )

    spec = TopologySpec(
        name=topology_data.get("name", "unnamed"),
        version=topology_data.get("version", "1.0"),
        description=topology_data.get("description", ""),
        agents=agents,
        edges=edges,
        metadata=topology_data.get("metadata", {}),
    )

    errors = spec.validate_edges()
    if errors:
        raise ValueError(f"Topology validation failed: {'; '.join(errors)}")

    return spec


def build_linear_topology(agent_ids: list[str], name: str = "linear") -> TopologySpec:
    """Factory: create a simple linear chain topology."""
    agents = [AgentNode(id=aid, name=aid) for aid in agent_ids]
    edges = [
        AgentEdge(source=agent_ids[i], target=agent_ids[i + 1])
        for i in range(len(agent_ids) - 1)
    ]
    return TopologySpec(name=name, agents=agents, edges=edges)


def build_star_topology(
    supervisor_id: str, worker_ids: list[str], name: str = "star"
) -> TopologySpec:
    """Factory: create a star topology with one supervisor and N workers."""
    agents = [AgentNode(id=supervisor_id, name=supervisor_id, role=AgentRole.SUPERVISOR)]
    agents.extend(AgentNode(id=wid, name=wid) for wid in worker_ids)
    edges = [AgentEdge(source=supervisor_id, target=wid) for wid in worker_ids]
    return TopologySpec(name=name, agents=agents, edges=edges)
