"""ExecutionTrace: full trace of a stress test run with time-travel navigation.

Records every step (LLM call, tool call, state mutation) in order,
enabling forward/backward stepping through execution for debugging.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TraceEventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    FAULT_INJECTED = "fault_injected"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    event_type: TraceEventType
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    step_index: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    fault_applied: str | None = None
    parent_event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class ExecutionTrace:
    """Full execution trace with time-travel navigation.

    Supports:
    - Sequential event recording
    - Forward/backward stepping (time-travel)
    - Filtering by agent, event type, or fault
    - Snapshot export for replay
    """

    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._events: list[TraceEvent] = []
        self._cursor: int = -1  # points to current position in time-travel
        self._start_time: float = time.time()

    def record(self, event: TraceEvent) -> int:
        """Record a new event. Returns the event index."""
        event_id = len(self._events)
        self._events.append(event)
        self._cursor = event_id
        return event_id

    def record_llm_call(
        self,
        agent_id: str,
        step_index: int,
        input_messages: list[dict] | None = None,
        output_text: str = "",
        fault_applied: str | None = None,
        tokens: dict[str, int] | None = None,
    ) -> int:
        return self.record(
            TraceEvent(
                event_type=TraceEventType.LLM_CALL,
                agent_id=agent_id,
                step_index=step_index,
                data={
                    "input_messages": input_messages or [],
                    "output_text": output_text,
                    "tokens": tokens or {},
                },
                fault_applied=fault_applied,
            )
        )

    def record_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        tool_output: Any = None,
        fault_applied: str | None = None,
    ) -> int:
        return self.record(
            TraceEvent(
                event_type=TraceEventType.TOOL_CALL,
                agent_id=agent_id,
                data={
                    "tool_name": tool_name,
                    "tool_input": tool_input or {},
                    "tool_output": tool_output,
                },
                fault_applied=fault_applied,
            )
        )

    def record_fault(self, agent_id: str, fault_type: str, details: dict[str, Any] | None = None) -> int:
        return self.record(
            TraceEvent(
                event_type=TraceEventType.FAULT_INJECTED,
                agent_id=agent_id,
                fault_applied=fault_type,
                data=details or {},
            )
        )

    # --- Time-travel navigation ---

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def current_event(self) -> TraceEvent | None:
        if 0 <= self._cursor < len(self._events):
            return self._events[self._cursor]
        return None

    def step_forward(self) -> TraceEvent | None:
        if self._cursor < len(self._events) - 1:
            self._cursor += 1
            return self._events[self._cursor]
        return None

    def step_backward(self) -> TraceEvent | None:
        if self._cursor > 0:
            self._cursor -= 1
            return self._events[self._cursor]
        return None

    def jump_to(self, index: int) -> TraceEvent | None:
        if 0 <= index < len(self._events):
            self._cursor = index
            return self._events[self._cursor]
        return None

    def rewind(self) -> None:
        self._cursor = 0 if self._events else -1

    # --- Filtering ---

    def get_agent_events(self, agent_id: str) -> list[TraceEvent]:
        return [e for e in self._events if e.agent_id == agent_id]

    def get_events_by_type(self, event_type: TraceEventType) -> list[TraceEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def get_fault_events(self) -> list[TraceEvent]:
        return [e for e in self._events if e.fault_applied is not None]

    def get_agent_timeline(self, agent_id: str) -> list[dict[str, Any]]:
        """Get a simplified timeline for a specific agent."""
        events = self.get_agent_events(agent_id)
        return [
            {
                "index": self._events.index(e),
                "type": e.event_type.value,
                "fault": e.fault_applied,
                "timestamp": e.timestamp,
            }
            for e in events
        ]

    # --- Export ---

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def summary(self) -> dict[str, Any]:
        agents = set(e.agent_id for e in self._events)
        return {
            "run_id": self.run_id,
            "total_events": len(self._events),
            "agents": sorted(agents),
            "llm_calls": len(self.get_events_by_type(TraceEventType.LLM_CALL)),
            "tool_calls": len(self.get_events_by_type(TraceEventType.TOOL_CALL)),
            "faults_injected": len(self.get_fault_events()),
            "duration_s": round(time.time() - self._start_time, 3),
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "events": [e.to_dict() for e in self._events],
                "summary": self.summary(),
            },
            indent=2,
            default=str,
        )

    @classmethod
    def from_json(cls, json_str: str) -> ExecutionTrace:
        data = json.loads(json_str)
        trace = cls(run_id=data.get("run_id", ""))
        for event_data in data.get("events", []):
            event = TraceEvent(
                event_type=TraceEventType(event_data["event_type"]),
                agent_id=event_data["agent_id"],
                timestamp=event_data.get("timestamp", 0.0),
                step_index=event_data.get("step_index", 0),
                data=event_data.get("data", {}),
                fault_applied=event_data.get("fault_applied"),
                parent_event_id=event_data.get("parent_event_id"),
            )
            trace.record(event)
        return trace
