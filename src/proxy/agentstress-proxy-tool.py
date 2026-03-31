"""Tool call proxy: intercepts and optionally faults agent tool invocations.

Wraps tool calls made by agents, allowing fault injection at the tool level
(e.g., simulate tool failures, inject wrong tool results, add latency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""

    agent_id: str
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    fault_applied: str | None = None
    timestamp: float = field(default_factory=time.time)


class ToolProxy:
    """Proxy wrapper for agent tool calls.

    Intercepts tool invocations to:
    1. Record all calls for telemetry
    2. Inject faults (failure, wrong output, latency)
    3. Support deterministic replay
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._records: list[ToolCallRecord] = []
        self._fault_handlers: dict[str, Callable] = {}
        self._global_fault: Callable | None = None

    def register_fault(self, tool_name: str, handler: Callable) -> None:
        """Register a fault handler for a specific tool."""
        self._fault_handlers[tool_name] = handler

    def set_global_fault(self, handler: Callable | None) -> None:
        """Set a fault handler that applies to all tool calls."""
        self._global_fault = handler

    def wrap_tool(self, tool_fn: Callable, tool_name: str) -> Callable:
        """Return a wrapped version of the tool function."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            record = ToolCallRecord(
                agent_id=self.agent_id,
                tool_name=tool_name,
                tool_input={"args": args, "kwargs": kwargs},
            )
            start = time.time()

            fault_handler = self._fault_handlers.get(tool_name, self._global_fault)

            try:
                if fault_handler:
                    result = fault_handler(tool_name, args, kwargs, tool_fn)
                    record.fault_applied = "custom_fault"
                else:
                    result = tool_fn(*args, **kwargs)

                record.tool_output = result
            except Exception as e:
                record.error = str(e)
                raise
            finally:
                record.latency_ms = (time.time() - start) * 1000
                self._records.append(record)

            return result

        wrapped.__name__ = f"proxied_{tool_name}"
        return wrapped

    @property
    def records(self) -> list[ToolCallRecord]:
        return list(self._records)

    def get_records_for_tool(self, tool_name: str) -> list[ToolCallRecord]:
        return [r for r in self._records if r.tool_name == tool_name]

    def reset(self) -> None:
        self._records.clear()
        self._fault_handlers.clear()
        self._global_fault = None
