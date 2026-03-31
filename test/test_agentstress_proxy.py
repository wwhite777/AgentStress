"""Tests for InterceptionPipeline and ProxiedChatModel."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from proxy.agentstress_proxy_intercept import (
    InterceptionContext,
    InterceptionPipeline,
    Interceptor,
)
from proxy.agentstress_proxy_llm import ProxiedChatModel
from conftest import StubChatModel


class CountingInterceptor:
    """Test interceptor that counts invocations."""

    def __init__(self, name: str = "counter") -> None:
        self._name = name
        self.before_count = 0
        self.after_count = 0

    @property
    def name(self) -> str:
        return self._name

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        self.before_count += 1
        ctx.metadata[f"{self._name}_before"] = self.before_count
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        self.after_count += 1
        ctx.metadata[f"{self._name}_after"] = self.after_count
        return ctx


class SkippingInterceptor:
    """Test interceptor that skips the LLM call."""

    @property
    def name(self) -> str:
        return "skipper"

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        ctx.skipped = True
        ctx.response = AIMessage(content="Skipped by test interceptor")
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        return ctx


class MessageModifyInterceptor:
    """Test interceptor that appends to messages."""

    @property
    def name(self) -> str:
        return "modifier"

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        ctx.messages.append(HumanMessage(content="[Injected by interceptor]"))
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        if ctx.response and isinstance(ctx.response.content, str):
            ctx.response = AIMessage(content=ctx.response.content + " [Modified]")
        return ctx


class TestAgentStressInterceptionPipeline:
    """Tests for the InterceptionPipeline."""

    def test_agentstress_pipeline_empty(self, agentstress_pipeline, agentstress_sample_context):
        ctx = agentstress_pipeline.execute(agentstress_sample_context)
        assert not ctx.skipped
        assert ctx.agent_id == "reasoning"

    def test_agentstress_pipeline_single_interceptor(self, agentstress_pipeline, agentstress_sample_context):
        counter = CountingInterceptor()
        agentstress_pipeline.add(counter)
        ctx = agentstress_pipeline.execute(agentstress_sample_context)
        assert counter.before_count == 1
        assert counter.after_count == 1

    def test_agentstress_pipeline_ordering(self, agentstress_pipeline, agentstress_sample_context):
        c1 = CountingInterceptor("first")
        c2 = CountingInterceptor("second")
        agentstress_pipeline.add(c1)
        agentstress_pipeline.add(c2)

        ctx = agentstress_pipeline.run_before(agentstress_sample_context)
        assert ctx.metadata["first_before"] == 1
        assert ctx.metadata["second_before"] == 1

    def test_agentstress_pipeline_skip_stops_before(self, agentstress_pipeline, agentstress_sample_context):
        skipper = SkippingInterceptor()
        counter = CountingInterceptor()
        agentstress_pipeline.add(skipper)
        agentstress_pipeline.add(counter)

        ctx = agentstress_pipeline.execute(agentstress_sample_context)
        assert ctx.skipped
        assert counter.before_count == 0  # skipped before reaching counter

    def test_agentstress_pipeline_remove(self, agentstress_pipeline):
        c1 = CountingInterceptor("keep")
        c2 = CountingInterceptor("remove")
        agentstress_pipeline.add(c1)
        agentstress_pipeline.add(c2)
        agentstress_pipeline.remove("remove")
        assert len(agentstress_pipeline.interceptors) == 1
        assert agentstress_pipeline.interceptors[0].name == "keep"

    def test_agentstress_pipeline_clear(self, agentstress_pipeline):
        agentstress_pipeline.add(CountingInterceptor())
        agentstress_pipeline.clear()
        assert len(agentstress_pipeline.interceptors) == 0

    def test_agentstress_pipeline_after_reverse_order(self, agentstress_pipeline, agentstress_sample_context):
        order = []

        class OrderTracker:
            def __init__(self, label):
                self._label = label

            @property
            def name(self):
                return self._label

            def before_call(self, ctx):
                return ctx

            def after_call(self, ctx):
                order.append(self._label)
                return ctx

        agentstress_pipeline.add(OrderTracker("A"))
        agentstress_pipeline.add(OrderTracker("B"))
        agentstress_pipeline.add(OrderTracker("C"))

        agentstress_pipeline.run_after(agentstress_sample_context)
        assert order == ["C", "B", "A"]


class TestAgentStressProxiedChatModel:
    """Tests for ProxiedChatModel."""

    def test_agentstress_proxied_llm_passthrough(self, agentstress_stub_llm, agentstress_pipeline):
        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        result = proxied.invoke([HumanMessage(content="Hello")])
        assert result.content == "Test LLM response."

    def test_agentstress_proxied_llm_with_interceptor(self, agentstress_stub_llm, agentstress_pipeline):
        counter = CountingInterceptor()
        agentstress_pipeline.add(counter)

        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        proxied.invoke([HumanMessage(content="Hello")])
        assert counter.before_count == 1
        assert counter.after_count == 1

    def test_agentstress_proxied_llm_skip(self, agentstress_stub_llm, agentstress_pipeline):
        agentstress_pipeline.add(SkippingInterceptor())

        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        result = proxied.invoke([HumanMessage(content="Hello")])
        assert result.content == "Skipped by test interceptor"

    def test_agentstress_proxied_llm_message_modification(self, agentstress_stub_llm, agentstress_pipeline):
        agentstress_pipeline.add(MessageModifyInterceptor())

        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        result = proxied.invoke([HumanMessage(content="Hello")])
        assert result.content == "Test LLM response. [Modified]"

    def test_agentstress_proxied_llm_type(self, agentstress_stub_llm, agentstress_pipeline):
        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        assert "proxied" in proxied._llm_type
        assert "stub" in proxied._llm_type

    def test_agentstress_proxied_llm_step_counter(self, agentstress_stub_llm, agentstress_pipeline):
        proxied = ProxiedChatModel(
            wrapped=agentstress_stub_llm,
            pipeline=agentstress_pipeline,
            agent_id="test-agent",
        )
        proxied.invoke([HumanMessage(content="Call 1")])
        proxied.invoke([HumanMessage(content="Call 2")])
        assert proxied.step_counter == 2
