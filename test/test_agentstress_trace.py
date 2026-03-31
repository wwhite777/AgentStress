"""Tests for ExecutionTrace with time-travel navigation."""

from __future__ import annotations

import json

import pytest

from telemetry.agentstress_telemetry_trace import (
    ExecutionTrace,
    TraceEvent,
    TraceEventType,
)


@pytest.fixture
def agentstress_trace():
    trace = ExecutionTrace(run_id="test-run-001")
    trace.record_llm_call("triage", step_index=1, output_text="Triage result")
    trace.record_llm_call("reasoning", step_index=2, output_text="Reasoning result", fault_applied="context_truncation")
    trace.record_tool_call("literature", tool_name="pubmed_search", tool_input={"query": "ACS"})
    trace.record_llm_call("report", step_index=3, output_text="SOAP note")
    trace.record_fault("reasoning", fault_type="context_noise", details={"noise_ratio": 0.15})
    return trace


class TestAgentStressTraceRecording:
    """Tests for event recording."""

    def test_agentstress_trace_record_events(self, agentstress_trace):
        assert len(agentstress_trace) == 5

    def test_agentstress_trace_llm_call_event(self, agentstress_trace):
        llm_events = agentstress_trace.get_events_by_type(TraceEventType.LLM_CALL)
        assert len(llm_events) == 3
        assert llm_events[0].agent_id == "triage"
        assert llm_events[0].data["output_text"] == "Triage result"

    def test_agentstress_trace_tool_call_event(self, agentstress_trace):
        tool_events = agentstress_trace.get_events_by_type(TraceEventType.TOOL_CALL)
        assert len(tool_events) == 1
        assert tool_events[0].data["tool_name"] == "pubmed_search"

    def test_agentstress_trace_fault_events(self, agentstress_trace):
        fault_events = agentstress_trace.get_fault_events()
        assert len(fault_events) == 2  # one on LLM call, one explicit fault record


class TestAgentStressTraceTimeTravel:
    """Tests for time-travel navigation."""

    def test_agentstress_trace_cursor_at_end(self, agentstress_trace):
        assert agentstress_trace.cursor == 4  # last event index

    def test_agentstress_trace_step_backward(self, agentstress_trace):
        event = agentstress_trace.step_backward()
        assert event is not None
        assert agentstress_trace.cursor == 3

    def test_agentstress_trace_step_forward_at_end(self, agentstress_trace):
        assert agentstress_trace.step_forward() is None
        assert agentstress_trace.cursor == 4

    def test_agentstress_trace_rewind(self, agentstress_trace):
        agentstress_trace.rewind()
        assert agentstress_trace.cursor == 0
        assert agentstress_trace.current_event.agent_id == "triage"

    def test_agentstress_trace_jump_to(self, agentstress_trace):
        event = agentstress_trace.jump_to(2)
        assert event is not None
        assert event.event_type == TraceEventType.TOOL_CALL

    def test_agentstress_trace_jump_out_of_bounds(self, agentstress_trace):
        assert agentstress_trace.jump_to(100) is None
        assert agentstress_trace.jump_to(-1) is None

    def test_agentstress_trace_full_traversal(self, agentstress_trace):
        agentstress_trace.rewind()
        events_visited = [agentstress_trace.current_event]
        while True:
            event = agentstress_trace.step_forward()
            if event is None:
                break
            events_visited.append(event)
        assert len(events_visited) == 5

    def test_agentstress_trace_backward_traversal(self, agentstress_trace):
        count = 0
        while agentstress_trace.step_backward() is not None:
            count += 1
        assert count == 4  # from index 4 back to 0


class TestAgentStressTraceFiltering:
    """Tests for event filtering."""

    def test_agentstress_trace_agent_events(self, agentstress_trace):
        reasoning_events = agentstress_trace.get_agent_events("reasoning")
        assert len(reasoning_events) == 2  # 1 LLM call + 1 fault

    def test_agentstress_trace_agent_timeline(self, agentstress_trace):
        timeline = agentstress_trace.get_agent_timeline("reasoning")
        assert len(timeline) == 2
        assert timeline[0]["type"] == "llm_call"
        assert timeline[0]["fault"] == "context_truncation"


class TestAgentStressTraceExport:
    """Tests for JSON export/import."""

    def test_agentstress_trace_summary(self, agentstress_trace):
        summary = agentstress_trace.summary()
        assert summary["total_events"] == 5
        assert summary["llm_calls"] == 3
        assert summary["tool_calls"] == 1
        assert summary["faults_injected"] == 2
        assert sorted(summary["agents"]) == ["literature", "reasoning", "report", "triage"]

    def test_agentstress_trace_to_json(self, agentstress_trace):
        json_str = agentstress_trace.to_json()
        data = json.loads(json_str)
        assert data["run_id"] == "test-run-001"
        assert len(data["events"]) == 5

    def test_agentstress_trace_roundtrip_json(self, agentstress_trace):
        json_str = agentstress_trace.to_json()
        restored = ExecutionTrace.from_json(json_str)

        assert len(restored) == len(agentstress_trace)
        assert restored.run_id == agentstress_trace.run_id

        for orig, rest in zip(agentstress_trace.events, restored.events):
            assert orig.event_type == rest.event_type
            assert orig.agent_id == rest.agent_id
            assert orig.fault_applied == rest.fault_applied

    def test_agentstress_trace_empty(self):
        trace = ExecutionTrace(run_id="empty")
        assert len(trace) == 0
        assert trace.current_event is None
        assert trace.step_forward() is None
        assert trace.step_backward() is None
