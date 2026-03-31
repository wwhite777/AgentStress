"""Tests for topology definition and parsing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from topology.agentstress_topology_define import (
    AgentEdge,
    AgentNode,
    AgentRole,
    EdgeType,
    TopologySpec,
)
from topology.agentstress_topology_parse import (
    build_linear_topology,
    build_star_topology,
    load_topology_yaml,
    parse_topology_dict,
)


class TestAgentStressTopologyDefine:
    """Tests for Pydantic topology models."""

    def test_agentstress_topology_agent_node_defaults(self):
        node = AgentNode(id="agent1", name="Agent 1")
        assert node.role == AgentRole.WORKER
        assert node.model == "default"
        assert not node.is_supervisor()

    def test_agentstress_topology_agent_node_supervisor(self):
        node = AgentNode(id="sup", name="Supervisor", role=AgentRole.SUPERVISOR)
        assert node.is_supervisor()

    def test_agentstress_topology_edge_defaults(self):
        edge = AgentEdge(source="a", target="b")
        assert edge.edge_type == EdgeType.DIRECTED
        assert edge.label == ""

    def test_agentstress_topology_spec_get_agent(self, agentstress_sample_topology):
        topo = agentstress_sample_topology
        agent = topo.get_agent("triage")
        assert agent is not None
        assert agent.name == "Triage"
        assert topo.get_agent("nonexistent") is None

    def test_agentstress_topology_spec_get_agent_ids(self, agentstress_sample_topology):
        ids = agentstress_sample_topology.get_agent_ids()
        assert ids == ["triage", "reasoning", "report"]

    def test_agentstress_topology_spec_get_neighbors(self, agentstress_sample_topology):
        topo = agentstress_sample_topology
        assert topo.get_neighbors("triage") == ["reasoning"]
        assert topo.get_neighbors("reasoning") == ["report"]
        assert topo.get_neighbors("report") == []

    def test_agentstress_topology_spec_get_upstream(self, agentstress_sample_topology):
        topo = agentstress_sample_topology
        assert topo.get_upstream("reasoning") == ["triage"]
        assert topo.get_upstream("report") == ["reasoning"]
        assert topo.get_upstream("triage") == []

    def test_agentstress_topology_spec_bidirectional_neighbors(self):
        topo = TopologySpec(
            name="bidir",
            agents=[
                AgentNode(id="a", name="A"),
                AgentNode(id="b", name="B"),
            ],
            edges=[
                AgentEdge(source="a", target="b", edge_type=EdgeType.BIDIRECTIONAL),
            ],
        )
        assert "b" in topo.get_neighbors("a")
        assert "a" in topo.get_neighbors("b")

    def test_agentstress_topology_spec_validate_edges_valid(self, agentstress_sample_topology):
        errors = agentstress_sample_topology.validate_edges()
        assert errors == []

    def test_agentstress_topology_spec_validate_edges_missing_source(self):
        topo = TopologySpec(
            name="bad",
            agents=[AgentNode(id="a", name="A")],
            edges=[AgentEdge(source="missing", target="a")],
        )
        errors = topo.validate_edges()
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_agentstress_topology_spec_validate_edges_self_loop(self):
        topo = TopologySpec(
            name="loop",
            agents=[AgentNode(id="a", name="A")],
            edges=[AgentEdge(source="a", target="a")],
        )
        errors = topo.validate_edges()
        assert any("Self-loop" in e for e in errors)


class TestAgentStressTopologyParse:
    """Tests for YAML loading and factory functions."""

    def test_agentstress_topology_parse_dict(self):
        raw = {
            "topology": {
                "name": "test",
                "agents": [
                    {"id": "a", "name": "Agent A", "role": "worker"},
                    {"id": "b", "name": "Agent B", "role": "supervisor"},
                ],
                "edges": [
                    {"source": "b", "target": "a", "edge_type": "directed"},
                ],
            }
        }
        topo = parse_topology_dict(raw)
        assert topo.name == "test"
        assert len(topo.agents) == 2
        assert len(topo.edges) == 1
        assert topo.get_agent("b").role == AgentRole.SUPERVISOR

    def test_agentstress_topology_parse_yaml_file(self):
        yaml_content = {
            "topology": {
                "name": "from-file",
                "agents": [
                    {"id": "x", "name": "X"},
                    {"id": "y", "name": "Y"},
                ],
                "edges": [
                    {"source": "x", "target": "y"},
                ],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            topo = load_topology_yaml(f.name)

        assert topo.name == "from-file"
        assert len(topo.agents) == 2

    def test_agentstress_topology_parse_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_topology_yaml("/nonexistent/path.yaml")

    def test_agentstress_topology_parse_invalid_edges(self):
        raw = {
            "topology": {
                "name": "bad",
                "agents": [{"id": "a", "name": "A"}],
                "edges": [{"source": "a", "target": "nonexistent"}],
            }
        }
        with pytest.raises(ValueError, match="validation failed"):
            parse_topology_dict(raw)

    def test_agentstress_topology_build_linear(self):
        topo = build_linear_topology(["a", "b", "c"])
        assert len(topo.agents) == 3
        assert len(topo.edges) == 2
        assert topo.get_neighbors("a") == ["b"]
        assert topo.get_neighbors("b") == ["c"]

    def test_agentstress_topology_build_star(self):
        topo = build_star_topology("supervisor", ["w1", "w2", "w3"])
        assert len(topo.agents) == 4
        assert len(topo.edges) == 3
        assert topo.get_agent("supervisor").role == AgentRole.SUPERVISOR
        assert set(topo.get_neighbors("supervisor")) == {"w1", "w2", "w3"}

    def test_agentstress_topology_load_example_config(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "agentstress-topology-example.yaml"
        if config_path.exists():
            topo = load_topology_yaml(config_path)
            assert topo.name == "medagent-clinical"
            assert len(topo.agents) == 4
            assert len(topo.edges) == 4
