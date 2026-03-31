"""Deterministic recording: capture full execution state for replay.

Records every LLM call (input messages, output, timing, fault state) plus
random seeds, enabling byte-identical replay of stress test runs.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxy.agentstress_proxy_intercept import InterceptionContext
from telemetry.agentstress_telemetry_trace import ExecutionTrace, TraceEventType


@dataclass
class RecordedStep:
    """A single recorded LLM call with full state for replay."""

    agent_id: str
    step_index: int
    input_messages: list[dict[str, Any]]
    output_content: str
    fault_applied: str | None = None
    random_seed: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "step_index": self.step_index,
            "input_messages": self.input_messages,
            "output_content": self.output_content,
            "fault_applied": self.fault_applied,
            "random_seed": self.random_seed,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecordedStep:
        return cls(
            agent_id=d["agent_id"],
            step_index=d["step_index"],
            input_messages=d.get("input_messages", []),
            output_content=d.get("output_content", ""),
            fault_applied=d.get("fault_applied"),
            random_seed=d.get("random_seed", 0),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Recording:
    """Complete recording of a stress test run."""

    run_id: str
    scenario_name: str = ""
    topology_name: str = ""
    master_seed: int = 0
    steps: list[RecordedStep] = field(default_factory=list)
    trace: ExecutionTrace | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return round(self.end_time - self.start_time, 3) if self.end_time else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "topology_name": self.topology_name,
            "master_seed": self.master_seed,
            "duration_s": self.duration_s,
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recording:
        rec = cls(
            run_id=d["run_id"],
            scenario_name=d.get("scenario_name", ""),
            topology_name=d.get("topology_name", ""),
            master_seed=d.get("master_seed", 0),
        )
        rec.steps = [RecordedStep.from_dict(s) for s in d.get("steps", [])]
        return rec


class ExecutionRecorder:
    """Interceptor that records every LLM call for deterministic replay.

    Captures input messages, output content, random state, and fault metadata.
    Sits at the end of the InterceptionPipeline (after faults and telemetry).
    """

    def __init__(self, run_id: str = "", master_seed: int | None = None) -> None:
        self.run_id = run_id or f"rec-{int(time.time())}"
        self.master_seed = master_seed if master_seed is not None else random.randint(0, 2**32 - 1)
        self._recording = Recording(
            run_id=self.run_id,
            master_seed=self.master_seed,
        )
        self._step_seed_counter = 0

    @property
    def name(self) -> str:
        return "execution-recorder"

    @property
    def recording(self) -> Recording:
        return self._recording

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        self._step_seed_counter += 1
        step_seed = (self.master_seed + self._step_seed_counter) % (2**32)
        ctx.metadata["replay_seed"] = step_seed
        ctx.metadata["replay_step_index"] = self._step_seed_counter
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        messages = []
        for msg in ctx.messages:
            messages.append({
                "role": msg.type if hasattr(msg, "type") else "unknown",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })

        output_content = ""
        if ctx.response and isinstance(ctx.response.content, str):
            output_content = ctx.response.content

        step = RecordedStep(
            agent_id=ctx.agent_id,
            step_index=ctx.step_index,
            input_messages=messages,
            output_content=output_content,
            fault_applied=ctx.fault_applied,
            random_seed=ctx.metadata.get("replay_seed", 0),
            timestamp=ctx.timestamp,
            metadata={
                k: v for k, v in ctx.metadata.items()
                if k not in ("replay_seed", "replay_step_index")
            },
        )
        self._recording.steps.append(step)
        return ctx

    def finalize(self) -> Recording:
        self._recording.end_time = time.time()
        return self._recording

    def save(self, path: str | Path) -> Path:
        """Save recording to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        recording = self.finalize()
        with open(path, "w") as f:
            json.dump(recording.to_dict(), f, indent=2, default=str)
        return path

    @staticmethod
    def load(path: str | Path) -> Recording:
        """Load a recording from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Recording file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return Recording.from_dict(data)

    def reset(self) -> None:
        self._recording = Recording(
            run_id=self.run_id,
            master_seed=self.master_seed,
        )
        self._step_seed_counter = 0
