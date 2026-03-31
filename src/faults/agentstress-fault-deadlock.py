"""Deadlock/livelock fault injectors: loop detection, token thrashing.

MRS analogy: Deadlock and livelock — agents get stuck in infinite
coordination loops or burn tokens without progress, draining the
API budget while producing no useful output.
"""

from __future__ import annotations

import random

from langchain_core.messages import AIMessage, HumanMessage

from faults.agentstress_fault_base import BaseFaultInjector, FaultConfig, FaultType
from proxy.agentstress_proxy_intercept import InterceptionContext


class DeadlockFault(BaseFaultInjector):
    """Simulate agent deadlock by forcing repeated delegation requests.

    MRS analogy: Deadlock — two or more agents wait for each other's output,
    creating a circular dependency that halts progress.

    The fault modifies the agent's response to request input from another agent,
    creating a delegation loop when applied to multiple agents in a cycle.

    Params:
        delegate_to (str): Agent ID to delegate to. Default "previous" (use upstream agent).
        delegation_message (str): Custom delegation text. Uses default if empty.
        max_loops (int): Maximum loops before the fault self-terminates. Default 5.
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.delegate_to = config.params.get("delegate_to", "previous")
        self.delegation_message = config.params.get("delegation_message", "")
        self.max_loops = config.params.get("max_loops", 5)
        self._loop_counts: dict[str, int] = {}

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        agent_key = ctx.agent_id
        self._loop_counts[agent_key] = self._loop_counts.get(agent_key, 0) + 1

        if self._loop_counts[agent_key] > self.max_loops:
            ctx.metadata["deadlock_terminated"] = True
            ctx.metadata["deadlock_loops"] = self._loop_counts[agent_key] - 1
            return ctx

        ctx.metadata["deadlock_loop_count"] = self._loop_counts[agent_key]
        ctx.metadata["deadlock_delegate_to"] = self.delegate_to
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        if ctx.fault_applied != FaultType.DEADLOCK.value:
            return ctx
        if ctx.metadata.get("deadlock_terminated"):
            return ctx

        delegate = self.delegate_to
        msg = self.delegation_message or (
            f"I need more information from {delegate} before I can proceed. "
            f"Requesting {delegate} to re-analyze the current state and provide updated findings."
        )

        ctx.response = AIMessage(content=msg)
        return ctx

    def reset(self) -> None:
        super().reset()
        self._loop_counts.clear()


class TokenThrashFault(BaseFaultInjector):
    """Simulate token thrashing by inflating context with verbose, repetitive content.

    MRS analogy: Livelock — the agent appears busy (generating tokens) but
    makes no meaningful progress, consuming budget without useful output.

    Params:
        inflation_factor (int): How many times to duplicate message content. Default 3.
        padding_text (str): Additional padding text to inject. Uses default if empty.
        target_input (bool): If True, inflate input messages. Default True.
        target_output (bool): If True, inflate output response. Default True.
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.inflation_factor = config.params.get("inflation_factor", 3)
        self.padding_text = config.params.get("padding_text", "")
        self.target_input = config.params.get("target_input", True)
        self.target_output = config.params.get("target_output", True)

    def _generate_padding(self) -> str:
        if self.padding_text:
            return self.padding_text

        fillers = [
            "Let me reconsider this from multiple angles to ensure thoroughness.",
            "To be comprehensive, I should also note the following considerations.",
            "Additionally, cross-referencing with related findings suggests further analysis.",
            "It is also worth mentioning several tangential but potentially relevant factors.",
            "For completeness, let me elaborate on the broader context of this analysis.",
        ]
        return " ".join(random.choices(fillers, k=self.inflation_factor))

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        if not self.target_input:
            return ctx

        padding = self._generate_padding()
        inflated = []
        for msg in ctx.messages:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                inflated_content = msg.content + ("\n" + padding) * self.inflation_factor
                inflated.append(msg.model_copy(update={"content": inflated_content}))
            else:
                inflated.append(msg)

        ctx.messages = inflated
        ctx.metadata["token_thrash_input_inflated"] = True
        ctx.metadata["token_thrash_inflation_factor"] = self.inflation_factor
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        if ctx.fault_applied != FaultType.TOKEN_THRASH.value:
            return ctx
        if not self.target_output:
            return ctx
        if not ctx.response or not isinstance(ctx.response.content, str):
            return ctx

        padding = self._generate_padding()
        inflated = ctx.response.content + ("\n" + padding) * self.inflation_factor
        ctx.response = AIMessage(content=inflated)
        ctx.metadata["token_thrash_output_inflated"] = True
        return ctx


DEADLOCK_FAULT_REGISTRY: dict[FaultType, type[BaseFaultInjector]] = {
    FaultType.DEADLOCK: DeadlockFault,
    FaultType.TOKEN_THRASH: TokenThrashFault,
}


def create_deadlock_fault(config: FaultConfig) -> BaseFaultInjector:
    """Factory function: create a deadlock/livelock fault injector from config."""
    cls = DEADLOCK_FAULT_REGISTRY.get(config.fault_type)
    if cls is None:
        raise ValueError(f"Unknown deadlock fault type: {config.fault_type}")
    return cls(config)
