"""Topology visualization: export to Mermaid and Graphviz DOT formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topology.agentstress_topology_define import AgentRole, EdgeType, TopologySpec


_ROLE_SHAPES = {
    AgentRole.SUPERVISOR: ("[[", "]]"),    # stadium
    AgentRole.ROUTER: ("{{", "}}"),        # hexagon
    AgentRole.WORKER: ("[", "]"),          # rectangle
    AgentRole.CRITIC: ("([", "])"),        # stadium alt
    AgentRole.AGGREGATOR: ("[(", ")]"),    # cylinder
}

_ROLE_COLORS = {
    AgentRole.SUPERVISOR: "#e94560",
    AgentRole.ROUTER: "#f39c12",
    AgentRole.WORKER: "#3498db",
    AgentRole.CRITIC: "#9b59b6",
    AgentRole.AGGREGATOR: "#2ecc71",
}

_DOT_ROLE_SHAPES = {
    AgentRole.SUPERVISOR: "doubleoctagon",
    AgentRole.ROUTER: "hexagon",
    AgentRole.WORKER: "box",
    AgentRole.CRITIC: "diamond",
    AgentRole.AGGREGATOR: "cylinder",
}


def to_mermaid(topology: TopologySpec, direction: str = "LR") -> str:
    """Export topology to Mermaid flowchart syntax.

    Args:
        topology: The topology to visualize.
        direction: Graph direction — "LR" (left-right) or "TD" (top-down).
    """
    lines = [f"graph {direction}"]

    for agent in topology.agents:
        open_s, close_s = _ROLE_SHAPES.get(agent.role, ("[", "]"))
        label = f"{agent.name}\\n({agent.role.value})"
        lines.append(f"    {agent.id}{open_s}\"{label}\"{close_s}")

    for edge in topology.edges:
        if edge.edge_type == EdgeType.BIDIRECTIONAL:
            arrow = "<-->"
        elif edge.edge_type == EdgeType.CONDITIONAL:
            arrow = "-.->|conditional|"
        else:
            arrow = "-->"

        if edge.label and edge.edge_type != EdgeType.CONDITIONAL:
            lines.append(f"    {edge.source} -->|\"{edge.label}\"| {edge.target}")
        else:
            lines.append(f"    {edge.source} {arrow} {edge.target}")

    for agent in topology.agents:
        color = _ROLE_COLORS.get(agent.role, "#95a5a6")
        lines.append(f"    style {agent.id} fill:{color},color:#fff,stroke:#333")

    return "\n".join(lines)


def to_graphviz(topology: TopologySpec, rankdir: str = "LR") -> str:
    """Export topology to Graphviz DOT syntax.

    Args:
        topology: The topology to visualize.
        rankdir: Graph direction — "LR" or "TB".
    """
    lines = [
        "digraph AgentStress {",
        f"    rankdir={rankdir};",
        '    node [fontname="Helvetica", fontsize=11, style=filled];',
        '    edge [fontname="Helvetica", fontsize=9];',
        "",
    ]

    for agent in topology.agents:
        shape = _DOT_ROLE_SHAPES.get(agent.role, "box")
        color = _ROLE_COLORS.get(agent.role, "#95a5a6")
        label = f"{agent.name}\\n({agent.role.value})"
        lines.append(
            f'    {agent.id} [label="{label}", shape={shape}, '
            f'fillcolor="{color}", fontcolor="white"];'
        )

    lines.append("")

    for edge in topology.edges:
        attrs = []
        if edge.label:
            attrs.append(f'label="{edge.label}"')
        if edge.edge_type == EdgeType.BIDIRECTIONAL:
            attrs.append("dir=both")
        elif edge.edge_type == EdgeType.CONDITIONAL:
            attrs.append("style=dashed")

        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f"    {edge.source} -> {edge.target}{attr_str};")

    lines.append("}")
    return "\n".join(lines)


def save_mermaid(topology: TopologySpec, path: str | Path, direction: str = "LR") -> Path:
    """Save Mermaid diagram to a .md file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = to_mermaid(topology, direction)
    with open(path, "w") as f:
        f.write(f"```mermaid\n{content}\n```\n")
    return path


def save_graphviz(topology: TopologySpec, path: str | Path, rankdir: str = "LR") -> Path:
    """Save Graphviz DOT to a .dot file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = to_graphviz(topology, rankdir)
    with open(path, "w") as f:
        f.write(content)
    return path
