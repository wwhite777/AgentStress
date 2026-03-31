"""Network fault injectors: message drop, API throttling.

MRS analogy: Packet loss and network latency — agents lose inter-agent
messages or experience degraded API responsiveness.
"""

from __future__ import annotations

import random
import time

from langchain_core.messages import AIMessage

from faults.agentstress_fault_base import BaseFaultInjector, FaultConfig, FaultType
from proxy.agentstress_proxy_intercept import InterceptionContext


class MessageDropFault(BaseFaultInjector):
    """Drop messages from the context to simulate inter-agent communication loss.

    MRS analogy: Packet loss — messages between agents are silently dropped,
    forcing downstream agents to operate with incomplete information.

    Params:
        drop_ratio (float): Fraction of non-system messages to drop. Default 0.3.
        drop_newest (bool): If True, drop newest messages first. Default False (random).
        min_keep (int): Minimum messages to keep. Default 1.
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.drop_ratio = config.params.get("drop_ratio", 0.3)
        self.drop_newest = config.params.get("drop_newest", False)
        self.min_keep = config.params.get("min_keep", 1)

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        from langchain_core.messages import SystemMessage

        system_msgs = [m for m in ctx.messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in ctx.messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.min_keep:
            return ctx

        num_drop = max(1, int(len(other_msgs) * self.drop_ratio))
        num_keep = max(self.min_keep, len(other_msgs) - num_drop)

        if self.drop_newest:
            kept = other_msgs[:num_keep]
        else:
            indices = sorted(random.sample(range(len(other_msgs)), num_keep))
            kept = [other_msgs[i] for i in indices]

        ctx.messages = system_msgs + kept
        ctx.metadata["messages_dropped"] = len(other_msgs) - len(kept)
        return ctx


class APIThrottleFault(BaseFaultInjector):
    """Simulate API rate limiting by injecting latency or returning error responses.

    MRS analogy: Network latency/congestion — the agent's connection to its
    LLM backend is degraded, causing timeouts or rate-limit errors.

    Params:
        mode (str): "latency" (add delay) or "error" (return 429-like response). Default "latency".
        latency_ms (float): Delay in milliseconds for latency mode. Default 2000.
        error_message (str): Error response text for error mode.
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.mode = config.params.get("mode", "latency")
        self.latency_ms = config.params.get("latency_ms", 2000.0)
        self.error_message = config.params.get(
            "error_message",
            "[API THROTTLED] Rate limit exceeded. Please retry after backoff.",
        )

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        if self.mode == "latency":
            delay_s = self.latency_ms / 1000.0
            time.sleep(delay_s)
            ctx.metadata["throttle_delay_ms"] = self.latency_ms
        elif self.mode == "error":
            ctx.skipped = True
            ctx.response = AIMessage(content=self.error_message)
            ctx.metadata["throttle_error"] = True
        return ctx


NETWORK_FAULT_REGISTRY: dict[FaultType, type[BaseFaultInjector]] = {
    FaultType.MESSAGE_DROP: MessageDropFault,
    FaultType.API_THROTTLE: APIThrottleFault,
}


def create_network_fault(config: FaultConfig) -> BaseFaultInjector:
    """Factory function: create a network fault injector from config."""
    cls = NETWORK_FAULT_REGISTRY.get(config.fault_type)
    if cls is None:
        raise ValueError(f"Unknown network fault type: {config.fault_type}")
    return cls(config)
