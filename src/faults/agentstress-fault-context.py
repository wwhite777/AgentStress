"""Context corruption fault injectors: truncation, noise injection, RAG failure."""

from __future__ import annotations

import random
import string

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from faults.agentstress_fault_base import BaseFaultInjector, FaultConfig, FaultType
from proxy.agentstress_proxy_intercept import InterceptionContext


class ContextTruncationFault(BaseFaultInjector):
    """Truncate the context window by removing older messages.

    MRS analogy: Sensor blindness — agent loses historical context,
    simulating memory corruption or context window overflow.

    Params:
        keep_ratio (float): Fraction of messages to keep (0.0-1.0). Default 0.3.
        keep_system (bool): Whether to preserve system messages. Default True.
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.keep_ratio = config.params.get("keep_ratio", 0.3)
        self.keep_system = config.params.get("keep_system", True)

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        if len(ctx.messages) <= 1:
            return ctx

        system_msgs = []
        other_msgs = []
        for msg in ctx.messages:
            if self.keep_system and isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        keep_count = max(1, int(len(other_msgs) * self.keep_ratio))
        truncated = other_msgs[-keep_count:]

        ctx.messages = system_msgs + truncated
        ctx.metadata["truncation_removed"] = len(other_msgs) - keep_count
        return ctx


class ContextNoiseFault(BaseFaultInjector):
    """Inject noise into message content.

    MRS analogy: Sensor noise — agent receives corrupted observations,
    simulating data corruption in inter-agent communication.

    Params:
        noise_ratio (float): Fraction of characters to corrupt. Default 0.1.
        target_role (str): Message role to corrupt ("human", "ai", "all"). Default "human".
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.noise_ratio = config.params.get("noise_ratio", 0.1)
        self.target_role = config.params.get("target_role", "human")

    def _corrupt_text(self, text: str) -> str:
        chars = list(text)
        num_corrupt = max(1, int(len(chars) * self.noise_ratio))
        positions = random.sample(range(len(chars)), min(num_corrupt, len(chars)))
        for pos in positions:
            chars[pos] = random.choice(string.ascii_letters + string.digits)
        return "".join(chars)

    def _should_corrupt_message(self, msg: BaseMessage) -> bool:
        if self.target_role == "all":
            return True
        if self.target_role == "human" and isinstance(msg, HumanMessage):
            return True
        if self.target_role == "ai" and isinstance(msg, AIMessage):
            return True
        return False

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        corrupted_count = 0
        new_messages = []
        for msg in ctx.messages:
            if self._should_corrupt_message(msg) and isinstance(msg.content, str):
                corrupted = msg.model_copy(update={"content": self._corrupt_text(msg.content)})
                new_messages.append(corrupted)
                corrupted_count += 1
            else:
                new_messages.append(msg)

        ctx.messages = new_messages
        ctx.metadata["noise_corrupted_messages"] = corrupted_count
        return ctx


class RAGFailureFault(BaseFaultInjector):
    """Simulate RAG retrieval failure by replacing retrieved content with empty or irrelevant data.

    MRS analogy: Sensor failure — agent's external knowledge source goes offline,
    forcing it to operate on internal knowledge alone.

    Params:
        failure_mode (str): "empty" (return nothing) or "irrelevant" (return garbage). Default "empty".
        rag_marker (str): String that identifies RAG content in messages. Default "[Retrieved Context]".
    """

    def __init__(self, config: FaultConfig) -> None:
        super().__init__(config)
        self.failure_mode = config.params.get("failure_mode", "empty")
        self.rag_marker = config.params.get("rag_marker", "[Retrieved Context]")

    def _generate_irrelevant_content(self) -> str:
        topics = [
            "The history of paperclip manufacturing dates back to 1867.",
            "Butterflies can taste with their feet.",
            "The average cloud weighs approximately 1.1 million pounds.",
            "Ancient Romans used crushed mouse brains as toothpaste.",
        ]
        return random.choice(topics)

    def apply_fault(self, ctx: InterceptionContext) -> InterceptionContext:
        new_messages = []
        rag_replaced = 0

        for msg in ctx.messages:
            if not isinstance(msg.content, str) or self.rag_marker not in msg.content:
                new_messages.append(msg)
                continue

            if self.failure_mode == "empty":
                replacement = msg.content.replace(
                    msg.content[msg.content.index(self.rag_marker):],
                    f"{self.rag_marker}\nNo results found.",
                )
            else:
                replacement = msg.content.replace(
                    msg.content[msg.content.index(self.rag_marker):],
                    f"{self.rag_marker}\n{self._generate_irrelevant_content()}",
                )

            new_messages.append(msg.model_copy(update={"content": replacement}))
            rag_replaced += 1

        ctx.messages = new_messages
        ctx.metadata["rag_failures_injected"] = rag_replaced
        return ctx


CONTEXT_FAULT_REGISTRY: dict[FaultType, type[BaseFaultInjector]] = {
    FaultType.CONTEXT_TRUNCATION: ContextTruncationFault,
    FaultType.CONTEXT_NOISE: ContextNoiseFault,
    FaultType.RAG_FAILURE: RAGFailureFault,
}


def create_context_fault(config: FaultConfig) -> BaseFaultInjector:
    """Factory function: create a context fault injector from config."""
    cls = CONTEXT_FAULT_REGISTRY.get(config.fault_type)
    if cls is None:
        raise ValueError(f"Unknown context fault type: {config.fault_type}")
    return cls(config)
