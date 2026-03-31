"""ProxiedChatModel: wraps any BaseChatModel with an InterceptionPipeline."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from proxy.agentstress_proxy_intercept import (
    InterceptionContext,
    InterceptionPipeline,
)


class ProxiedChatModel(BaseChatModel):
    """Drop-in replacement for any BaseChatModel that routes calls through an InterceptionPipeline.

    The pipeline interceptors can modify input messages (before_call),
    modify the response (after_call), or skip the call entirely.
    """

    model_config = {"arbitrary_types_allowed": True}

    wrapped: BaseChatModel
    pipeline: InterceptionPipeline
    agent_id: str = "unknown"
    step_counter: int = 0

    @property
    def _llm_type(self) -> str:
        return f"proxied-{self.wrapped._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "wrapped": self.wrapped._identifying_params,
            "agent_id": self.agent_id,
            "pipeline_interceptors": [i.name for i in self.pipeline.interceptors],
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.step_counter += 1

        ctx = InterceptionContext(
            agent_id=self.agent_id,
            messages=list(messages),
            call_kwargs={"stop": stop, **kwargs},
            step_index=self.step_counter,
        )

        ctx = self.pipeline.run_before(ctx)

        if ctx.skipped:
            fallback = ctx.response or AIMessage(content="[SKIPPED BY FAULT INJECTOR]")
            return ChatResult(generations=[ChatGeneration(message=fallback)])

        result = self.wrapped._generate(
            ctx.messages,
            stop=ctx.call_kwargs.get("stop"),
            run_manager=run_manager,
            **{k: v for k, v in ctx.call_kwargs.items() if k != "stop"},
        )

        if result.generations:
            ctx.response = result.generations[0].message

        ctx = self.pipeline.run_after(ctx)

        if ctx.response and ctx.response != result.generations[0].message:
            return ChatResult(generations=[ChatGeneration(message=ctx.response)])

        return result
