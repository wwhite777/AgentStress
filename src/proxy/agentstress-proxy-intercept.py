"""InterceptionPipeline: ordered chain of interceptors for LLM call proxying."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@dataclass
class InterceptionContext:
    """Mutable context passed through the interception pipeline."""

    agent_id: str
    messages: list[BaseMessage]
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    response: BaseMessage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    step_index: int = 0
    fault_applied: str | None = None
    skipped: bool = False


@runtime_checkable
class Interceptor(Protocol):
    """Protocol for pipeline interceptors (faults, telemetry, recorders)."""

    @property
    def name(self) -> str: ...

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        """Process context before the LLM call. May modify messages or skip the call."""
        ...

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        """Process context after the LLM call. May modify the response."""
        ...


class InterceptionPipeline:
    """Ordered chain of interceptors applied to every LLM call.

    Pipeline order: faults → telemetry → recording (before_call)
    Reverse order for after_call.
    """

    def __init__(self) -> None:
        self._interceptors: list[Interceptor] = []

    def add(self, interceptor: Interceptor) -> None:
        self._interceptors.append(interceptor)

    def remove(self, name: str) -> None:
        self._interceptors = [i for i in self._interceptors if i.name != name]

    def clear(self) -> None:
        self._interceptors.clear()

    @property
    def interceptors(self) -> list[Interceptor]:
        return list(self._interceptors)

    def run_before(self, ctx: InterceptionContext) -> InterceptionContext:
        """Run all before_call interceptors in order."""
        for interceptor in self._interceptors:
            ctx = interceptor.before_call(ctx)
            if ctx.skipped:
                break
        return ctx

    def run_after(self, ctx: InterceptionContext) -> InterceptionContext:
        """Run all after_call interceptors in reverse order."""
        for interceptor in reversed(self._interceptors):
            ctx = interceptor.after_call(ctx)
        return ctx

    def execute(self, ctx: InterceptionContext) -> InterceptionContext:
        """Run full before → (caller handles LLM) → after cycle.

        Note: The actual LLM call happens in ProxiedChatModel between
        run_before() and run_after(). This method is for testing pipelines
        without a real LLM.
        """
        ctx = self.run_before(ctx)
        if not ctx.skipped:
            ctx = self.run_after(ctx)
        return ctx
