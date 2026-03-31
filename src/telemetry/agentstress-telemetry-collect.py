"""Telemetry collection: StepMetrics and TelemetryCollector interceptor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from proxy.agentstress_proxy_intercept import InterceptionContext


@dataclass
class StepMetrics:
    """Metrics captured for a single LLM call step."""

    agent_id: str
    step_index: int
    timestamp: float
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    input_messages_count: int = 0
    fault_applied: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TelemetryCollector:
    """Interceptor that records StepMetrics for every LLM call.

    Sits in the InterceptionPipeline after fault injectors.
    Captures timing, token counts, and fault metadata.
    """

    def __init__(self) -> None:
        self._steps: list[StepMetrics] = []
        self._call_start_times: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "telemetry-collector"

    @property
    def steps(self) -> list[StepMetrics]:
        return list(self._steps)

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        call_key = f"{ctx.agent_id}:{ctx.step_index}"
        self._call_start_times[call_key] = time.time()

        ctx.metadata["telemetry_input_messages"] = len(ctx.messages)
        input_chars = sum(len(m.content) if isinstance(m.content, str) else 0 for m in ctx.messages)
        ctx.metadata["telemetry_input_chars"] = input_chars
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        call_key = f"{ctx.agent_id}:{ctx.step_index}"
        start_time = self._call_start_times.pop(call_key, ctx.timestamp)
        latency_ms = (time.time() - start_time) * 1000

        input_chars = ctx.metadata.get("telemetry_input_chars", 0)
        output_chars = 0
        if ctx.response and isinstance(ctx.response.content, str):
            output_chars = len(ctx.response.content)

        input_tokens = self._estimate_tokens(input_chars)
        output_tokens = self._estimate_tokens(output_chars)

        if ctx.response and hasattr(ctx.response, "usage_metadata") and ctx.response.usage_metadata:
            usage = ctx.response.usage_metadata
            if hasattr(usage, "input_tokens") and usage.input_tokens:
                input_tokens = usage.input_tokens
            if hasattr(usage, "output_tokens") and usage.output_tokens:
                output_tokens = usage.output_tokens

        step = StepMetrics(
            agent_id=ctx.agent_id,
            step_index=ctx.step_index,
            timestamp=ctx.timestamp,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_messages_count=ctx.metadata.get("telemetry_input_messages", 0),
            fault_applied=ctx.fault_applied,
            metadata={k: v for k, v in ctx.metadata.items() if not k.startswith("telemetry_")},
        )

        self._steps.append(step)
        return ctx

    def _estimate_tokens(self, char_count: int) -> int:
        """Rough estimate: ~4 chars per token for English text."""
        return max(1, char_count // 4) if char_count > 0 else 0

    def get_agent_steps(self, agent_id: str) -> list[StepMetrics]:
        return [s for s in self._steps if s.agent_id == agent_id]

    def get_total_tokens(self) -> int:
        return sum(s.total_tokens for s in self._steps)

    def get_total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self._steps)

    def get_fault_count(self) -> int:
        return sum(1 for s in self._steps if s.fault_applied is not None)

    def summary(self) -> dict:
        return {
            "total_steps": len(self._steps),
            "total_tokens": self.get_total_tokens(),
            "total_latency_ms": round(self.get_total_latency_ms(), 2),
            "faults_triggered": self.get_fault_count(),
            "agents_observed": list({s.agent_id for s in self._steps}),
        }

    def reset(self) -> None:
        self._steps.clear()
        self._call_start_times.clear()
