"""Base fault injector ABC and FaultConfig."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from proxy.agentstress_proxy_intercept import InterceptionContext


class FaultType(str, Enum):
    CONTEXT_CORRUPTION = "context_corruption"
    CONTEXT_TRUNCATION = "context_truncation"
    CONTEXT_NOISE = "context_noise"
    RAG_FAILURE = "rag_failure"
    MESSAGE_DROP = "message_drop"
    API_THROTTLE = "api_throttle"
    BYZANTINE = "byzantine"
    HALLUCINATION = "hallucination"
    DEADLOCK = "deadlock"
    TOKEN_THRASH = "token_thrash"


class FaultSchedule(str, Enum):
    CONTINUOUS = "continuous"
    BURST = "burst"
    PROGRESSIVE = "progressive"
    ONCE = "once"


class FaultConfig(BaseModel):
    """Configuration for a single fault injection."""

    fault_type: FaultType
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    target_agents: list[str] = Field(
        default_factory=list,
        description="Agent IDs to target. Empty = all agents.",
    )
    schedule: FaultSchedule = Field(default=FaultSchedule.CONTINUOUS)
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    def should_target(self, agent_id: str) -> bool:
        if not self.enabled:
            return False
        if not self.target_agents:
            return True
        return agent_id in self.target_agents


class BaseFaultInjector(ABC):
    """Abstract base for all fault injectors.

    Implements the Interceptor protocol from InterceptionPipeline.
    """

    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self._trigger_count = 0

    @property
    def name(self) -> str:
        return f"fault-{self.config.fault_type.value}"

    @abstractmethod
    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        """Apply the fault to the interception context."""
        ...

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        if not self.config.should_target(ctx.agent_id):
            return ctx

        import random

        if random.random() > self.config.probability:
            return ctx

        self._trigger_count += 1
        ctx.fault_applied = self.config.fault_type.value
        ctx.metadata["fault_trigger_count"] = self._trigger_count
        return self.apply_fault(ctx)

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        return ctx

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def reset(self) -> None:
        self._trigger_count = 0
