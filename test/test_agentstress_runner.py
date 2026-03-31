"""Tests for runner modules (scenario loader, engine, report)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from faults.agentstress_fault_base import FaultConfig, FaultSchedule, FaultType
from faults.agentstress_fault_context import ContextTruncationFault
from faults.agentstress_fault_schedule import ScheduledFaultWrapper
from runner.agentstress_runner_scenario import (
    EvalConfig,
    ScenarioSpec,
    SweepConfig,
    instantiate_faults,
    load_scenario_yaml,
    parse_scenario_dict,
)
from runner.agentstress_runner_report import generate_html_report, generate_json_report


# --- ScenarioSpec loader ---

class TestAgentStressScenarioLoader:

    def test_agentstress_scenario_parse_dict(self):
        raw = {
            "scenario": {
                "name": "test-scenario",
                "faults": [
                    {
                        "fault_type": "context_truncation",
                        "probability": 0.8,
                        "target_agents": ["reasoning"],
                        "schedule": "continuous",
                        "params": {"keep_ratio": 0.3},
                    }
                ],
                "sweep": {"enabled": True, "values": [0.1, 0.5, 1.0]},
            }
        }
        spec = parse_scenario_dict(raw)
        assert spec.name == "test-scenario"
        assert len(spec.fault_configs) == 1
        assert spec.fault_configs[0].fault_type == FaultType.CONTEXT_TRUNCATION
        assert spec.fault_configs[0].probability == 0.8
        assert spec.sweep.enabled
        assert len(spec.sweep.values) == 3

    def test_agentstress_scenario_parse_yaml_file(self):
        yaml_content = {
            "scenario": {
                "name": "from-file",
                "faults": [
                    {"fault_type": "context_noise", "probability": 0.5}
                ],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            spec = load_scenario_yaml(f.name)

        assert spec.name == "from-file"
        assert len(spec.fault_configs) == 1

    def test_agentstress_scenario_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenario_yaml("/nonexistent.yaml")

    def test_agentstress_scenario_multiple_faults(self):
        raw = {
            "scenario": {
                "name": "multi",
                "faults": [
                    {"fault_type": "context_truncation", "params": {"keep_ratio": 0.3}},
                    {"fault_type": "message_drop", "params": {"drop_ratio": 0.5}},
                    {"fault_type": "byzantine"},
                ],
            }
        }
        spec = parse_scenario_dict(raw)
        assert len(spec.fault_configs) == 3

    def test_agentstress_scenario_eval_config(self):
        raw = {
            "scenario": {
                "name": "eval-test",
                "faults": [],
                "evaluation": {
                    "judge_model": "my-model",
                    "metrics": ["quality", "safety"],
                },
            }
        }
        spec = parse_scenario_dict(raw)
        assert spec.evaluation.judge_model == "my-model"
        assert "quality" in spec.evaluation.metrics

    def test_agentstress_scenario_to_dict(self):
        spec = ScenarioSpec(
            name="test",
            fault_configs=[
                FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION, probability=0.5)
            ],
        )
        d = spec.to_dict()
        assert d["name"] == "test"
        assert len(d["faults"]) == 1

    def test_agentstress_scenario_load_example_configs(self):
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        for yaml_file in configs_dir.glob("agentstress-scenario-*.yaml"):
            spec = load_scenario_yaml(yaml_file)
            assert spec.name
            assert len(spec.fault_configs) > 0


# --- Instantiate faults ---

class TestAgentStressInstantiateFaults:

    def test_agentstress_instantiate_continuous(self):
        spec = ScenarioSpec(
            name="test",
            fault_configs=[
                FaultConfig(
                    fault_type=FaultType.CONTEXT_TRUNCATION,
                    schedule=FaultSchedule.CONTINUOUS,
                    params={"keep_ratio": 0.5},
                )
            ],
        )
        faults = instantiate_faults(spec)
        assert len(faults) == 1
        assert isinstance(faults[0], ContextTruncationFault)

    def test_agentstress_instantiate_scheduled(self):
        spec = ScenarioSpec(
            name="test",
            fault_configs=[
                FaultConfig(
                    fault_type=FaultType.CONTEXT_TRUNCATION,
                    schedule=FaultSchedule.BURST,
                    params={"keep_ratio": 0.5},
                )
            ],
        )
        faults = instantiate_faults(spec)
        assert len(faults) == 1
        assert isinstance(faults[0], ScheduledFaultWrapper)

    def test_agentstress_instantiate_probability_override(self):
        spec = ScenarioSpec(
            name="test",
            fault_configs=[
                FaultConfig(
                    fault_type=FaultType.CONTEXT_TRUNCATION,
                    probability=0.5,
                    params={"keep_ratio": 0.3},
                )
            ],
        )
        faults = instantiate_faults(spec, probability_override=0.9)
        assert faults[0].config.probability == 0.9

    def test_agentstress_instantiate_all_fault_types(self):
        configs = [
            FaultConfig(fault_type=FaultType.CONTEXT_TRUNCATION, params={"keep_ratio": 0.5}),
            FaultConfig(fault_type=FaultType.CONTEXT_NOISE, params={"noise_ratio": 0.1}),
            FaultConfig(fault_type=FaultType.RAG_FAILURE),
            FaultConfig(fault_type=FaultType.MESSAGE_DROP),
            FaultConfig(fault_type=FaultType.API_THROTTLE, params={"mode": "latency", "latency_ms": 10}),
            FaultConfig(fault_type=FaultType.BYZANTINE),
            FaultConfig(fault_type=FaultType.HALLUCINATION),
            FaultConfig(fault_type=FaultType.DEADLOCK, params={"max_loops": 2}),
            FaultConfig(fault_type=FaultType.TOKEN_THRASH),
        ]
        spec = ScenarioSpec(name="all", fault_configs=configs)
        faults = instantiate_faults(spec)
        assert len(faults) == 9


# --- Report generation ---

class TestAgentStressReportGeneration:

    def test_agentstress_report_json(self, tmp_path):
        data = {
            "scenario_name": "test",
            "baseline_score": 0.9,
            "stressed_score": 0.6,
        }

        class FakeResult:
            def to_dict(self):
                return data

        path = generate_json_report(FakeResult(), tmp_path / "report.json")
        assert path.exists()

        with open(path) as f:
            report = json.load(f)
        assert report["result"]["scenario_name"] == "test"
        assert "generated_at" in report

    def test_agentstress_report_html(self, tmp_path):
        data = {
            "scenario_name": "test-html",
            "topology_name": "medagent",
            "baseline_score": 0.9,
            "stressed_score": 0.6,
            "metrics": {
                "quality": {
                    "baseline_score": 0.9,
                    "stressed_score": 0.6,
                    "score_delta": -0.3,
                    "degradation_pct": 33.33,
                    "criterion_deltas": {"quality": -0.2, "safety": -0.1},
                },
                "cost": {
                    "baseline_usd": 0.01,
                    "stressed_usd": 0.03,
                    "overhead_ratio": 3.0,
                },
            },
        }

        class FakeResult:
            def to_dict(self):
                return data

        path = generate_html_report(FakeResult(), tmp_path / "report.html")
        assert path.exists()

        html = path.read_text()
        assert "AgentStress Report" in html
        assert "test-html" in html
        assert "33.33" in html

    def test_agentstress_report_html_with_blast(self, tmp_path):
        data = {
            "scenario_name": "blast-test",
            "topology_name": "test-topo",
            "baseline_score": 0.9,
            "stressed_score": None,
            "blast_radius": {
                "critical_agents": ["triage"],
                "redundant_agents": ["report"],
                "system_resilience_avg_degradation_pct": 40.0,
                "agent_results": [
                    {
                        "agent_id": "triage",
                        "agent_role": "router",
                        "degraded_score": 0.2,
                        "degradation_pct": 77.8,
                        "criticality": "critical",
                        "affected_downstream": ["reasoning"],
                    },
                ],
            },
        }

        class FakeResult:
            def to_dict(self):
                return data

        path = generate_html_report(FakeResult(), tmp_path / "blast.html")
        html = path.read_text()
        assert "Blast Radius" in html
        assert "triage" in html
        assert "critical" in html

    def test_agentstress_report_json_creates_dirs(self, tmp_path):
        path = generate_json_report({"test": True}, tmp_path / "sub" / "dir" / "report.json")
        assert path.exists()
