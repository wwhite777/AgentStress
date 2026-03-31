"""Inter-agent state proxy: intercepts and optionally faults shared state mutations.

In multi-agent frameworks like LangGraph, agents communicate via a shared
state dict. This proxy wraps state access to enable fault injection at
the state layer (e.g., corrupt state fields, drop updates, inject stale data).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StateAccessRecord:
    """Record of a single state read or write operation."""

    agent_id: str
    operation: str  # "read" or "write"
    key: str
    value_snapshot: Any = None
    fault_applied: str | None = None
    timestamp: float = field(default_factory=time.time)


class StateProxy:
    """Proxy for inter-agent shared state access.

    Wraps a state dict to intercept reads and writes, enabling:
    1. Telemetry on state access patterns
    2. Fault injection (corrupt values, drop writes, inject stale data)
    3. State snapshot recording for replay
    """

    def __init__(self, state: dict[str, Any], agent_id: str) -> None:
        self._state = state
        self.agent_id = agent_id
        self._records: list[StateAccessRecord] = []
        self._read_faults: dict[str, Callable] = {}
        self._write_faults: dict[str, Callable] = {}
        self._frozen_keys: set[str] = set()

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from shared state, with optional fault injection."""
        value = self._state.get(key, default)

        record = StateAccessRecord(
            agent_id=self.agent_id,
            operation="read",
            key=key,
        )

        if key in self._read_faults:
            value = self._read_faults[key](key, value)
            record.fault_applied = "read_fault"

        try:
            record.value_snapshot = copy.deepcopy(value)
        except (TypeError, copy.Error):
            record.value_snapshot = str(value)

        self._records.append(record)
        return value

    def write(self, key: str, value: Any) -> None:
        """Write a value to shared state, with optional fault injection."""
        record = StateAccessRecord(
            agent_id=self.agent_id,
            operation="write",
            key=key,
        )

        if key in self._frozen_keys:
            record.fault_applied = "write_blocked_frozen"
            self._records.append(record)
            return  # silently drop the write

        if key in self._write_faults:
            value = self._write_faults[key](key, value)
            record.fault_applied = "write_fault"

        try:
            record.value_snapshot = copy.deepcopy(value)
        except (TypeError, copy.Error):
            record.value_snapshot = str(value)

        self._state[key] = value
        self._records.append(record)

    def register_read_fault(self, key: str, handler: Callable) -> None:
        """Register a fault handler for reads on a specific key."""
        self._read_faults[key] = handler

    def register_write_fault(self, key: str, handler: Callable) -> None:
        """Register a fault handler for writes on a specific key."""
        self._write_faults[key] = handler

    def freeze_key(self, key: str) -> None:
        """Freeze a key so writes are silently dropped (stale data fault)."""
        self._frozen_keys.add(key)

    def unfreeze_key(self, key: str) -> None:
        self._frozen_keys.discard(key)

    @property
    def records(self) -> list[StateAccessRecord]:
        return list(self._records)

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def get_write_records(self) -> list[StateAccessRecord]:
        return [r for r in self._records if r.operation == "write"]

    def get_read_records(self) -> list[StateAccessRecord]:
        return [r for r in self._records if r.operation == "read"]

    def reset(self) -> None:
        self._records.clear()
        self._read_faults.clear()
        self._write_faults.clear()
        self._frozen_keys.clear()
