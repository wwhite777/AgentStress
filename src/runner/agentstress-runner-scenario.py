"""ScenarioSpec loader: parse scenario YAML and instantiate fault injectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from faults.agentstress_fault_base import BaseFaultInjector, FaultConfig, FaultSchedule, FaultType
from faults.agentstress_fault_context import create_context_fault
from faults.agentstress_fault_network import create_network_fault
from faults.agentstress_fault_byzantine import create_byzantine_fault
from faults.agentstress_fault_deadlock import create_deadlock_fault
from faults.agentstress_fault_schedule import wrap_with_schedule


_CONTEXT_FAULTS = {FaultType.CONTEXT_TRUNCATION, FaultType.CONTEXT_NOISE, FaultType.RAG_FAILURE}
_NETWORK_FAULTS = {FaultType.MESSAGE_DROP, FaultType.API_THROTTLE}
_BYZANTINE_FAULTS = {FaultType.BYZANTINE, FaultType.HALLUCINATION}
_DEADLOCK_FAULTS = {FaultType.DEADLOCK, FaultType.TOKEN_THRASH}


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps."""

    enabled: bool = False
    parameter: str = "probability"
    values: list[float] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    judge_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    metrics: list[str] = field(default_factory=lambda: ["task_completion", "output_quality"])
    judge_base_url: str = "http://localhost:8000/v1"


@dataclass
class ScenarioSpec:
    """Parsed scenario specification."""

    name: str
    version: str = "1.0"
    description: str = ""
    fault_configs: list[FaultConfig] = field(default_factory=list)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "faults": [
                {
                    "fault_type": fc.fault_type.value,
                    "probability": fc.probability,
                    "target_agents": fc.target_agents,
                    "schedule": fc.schedule.value,
                    "params": fc.params,
                }
                for fc in self.fault_configs
            ],
            "sweep": {"enabled": self.sweep.enabled, "values": self.sweep.values},
            "metadata": self.metadata,
        }


def _create_fault(config: FaultConfig) -> BaseFaultInjector:
    """Route to the correct fault factory based on type."""
    ft = config.fault_type
    if ft in _CONTEXT_FAULTS:
        return create_context_fault(config)
    if ft in _NETWORK_FAULTS:
        return create_network_fault(config)
    if ft in _BYZANTINE_FAULTS:
        return create_byzantine_fault(config)
    if ft in _DEADLOCK_FAULTS:
        return create_deadlock_fault(config)
    raise ValueError(f"Unknown fault type: {ft}")


def load_scenario_yaml(path: str | Path) -> ScenarioSpec:
    """Load a ScenarioSpec from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return parse_scenario_dict(raw)


def parse_scenario_dict(raw: dict[str, Any]) -> ScenarioSpec:
    """Parse a raw dict into a ScenarioSpec."""
    scenario_data = raw.get("scenario", raw)

    fault_configs = []
    for fault_raw in scenario_data.get("faults", []):
        fault_configs.append(
            FaultConfig(
                fault_type=FaultType(fault_raw["fault_type"]),
                probability=fault_raw.get("probability", 1.0),
                target_agents=fault_raw.get("target_agents", []),
                schedule=FaultSchedule(fault_raw.get("schedule", "continuous")),
                params=fault_raw.get("params", {}),
            )
        )

    sweep_raw = scenario_data.get("sweep", {})
    sweep = SweepConfig(
        enabled=sweep_raw.get("enabled", False),
        parameter=sweep_raw.get("parameter", "probability"),
        values=sweep_raw.get("values", []),
    )

    eval_raw = scenario_data.get("evaluation", {})
    evaluation = EvalConfig(
        judge_model=eval_raw.get("judge_model", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
        metrics=eval_raw.get("metrics", ["task_completion", "output_quality"]),
        judge_base_url=eval_raw.get("judge_base_url", "http://localhost:8000/v1"),
    )

    return ScenarioSpec(
        name=scenario_data.get("name", "unnamed"),
        version=scenario_data.get("version", "1.0"),
        description=scenario_data.get("description", ""),
        fault_configs=fault_configs,
        sweep=sweep,
        evaluation=evaluation,
        metadata=scenario_data.get("metadata", {}),
    )


def instantiate_faults(
    scenario: ScenarioSpec,
    probability_override: float | None = None,
) -> list:
    """Create fault injector instances from a scenario spec.

    Returns list of ScheduledFaultWrapper (or raw BaseFaultInjector for continuous).
    """
    injectors = []
    for config in scenario.fault_configs:
        if probability_override is not None:
            config = config.model_copy(update={"probability": probability_override})

        fault = _create_fault(config)

        if config.schedule != FaultSchedule.CONTINUOUS:
            injectors.append(wrap_with_schedule(fault))
        else:
            injectors.append(fault)

    return injectors
