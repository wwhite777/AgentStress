"""Tests for replay module (recording + playback)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from proxy.agentstress_proxy_intercept import InterceptionContext
from replay.agentstress_replay_record import ExecutionRecorder, Recording, RecordedStep
from replay.agentstress_replay_player import ReplayPlayer


# --- RecordedStep ---

class TestAgentStressRecordedStep:

    def test_agentstress_recorded_step_roundtrip(self):
        step = RecordedStep(
            agent_id="reasoning",
            step_index=1,
            input_messages=[{"role": "human", "content": "Hello"}],
            output_content="Response text",
            fault_applied="context_truncation",
            random_seed=42,
        )
        d = step.to_dict()
        restored = RecordedStep.from_dict(d)
        assert restored.agent_id == "reasoning"
        assert restored.output_content == "Response text"
        assert restored.fault_applied == "context_truncation"
        assert restored.random_seed == 42


# --- Recording ---

class TestAgentStressRecording:

    def test_agentstress_recording_roundtrip(self):
        rec = Recording(run_id="test-001", scenario_name="test", master_seed=42)
        rec.steps.append(RecordedStep(
            agent_id="a", step_index=1,
            input_messages=[], output_content="out1",
        ))
        rec.steps.append(RecordedStep(
            agent_id="b", step_index=2,
            input_messages=[], output_content="out2",
        ))

        d = rec.to_dict()
        restored = Recording.from_dict(d)
        assert restored.run_id == "test-001"
        assert len(restored.steps) == 2
        assert restored.master_seed == 42


# --- ExecutionRecorder ---

class TestAgentStressExecutionRecorder:

    def test_agentstress_recorder_captures_steps(self):
        recorder = ExecutionRecorder(run_id="test", master_seed=42)

        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Input data")],
            step_index=1,
        )
        ctx = recorder.before_call(ctx)
        assert "replay_seed" in ctx.metadata

        ctx.response = AIMessage(content="Analysis result")
        ctx = recorder.after_call(ctx)

        assert len(recorder.recording.steps) == 1
        assert recorder.recording.steps[0].agent_id == "reasoning"
        assert recorder.recording.steps[0].output_content == "Analysis result"

    def test_agentstress_recorder_multiple_steps(self):
        recorder = ExecutionRecorder(run_id="multi", master_seed=100)

        for i, agent_id in enumerate(["triage", "reasoning", "report"]):
            ctx = InterceptionContext(
                agent_id=agent_id,
                messages=[HumanMessage(content=f"Input {i}")],
                step_index=i,
            )
            ctx = recorder.before_call(ctx)
            ctx.response = AIMessage(content=f"Output {i}")
            recorder.after_call(ctx)

        assert len(recorder.recording.steps) == 3
        assert recorder.recording.steps[2].agent_id == "report"

    def test_agentstress_recorder_captures_faults(self):
        recorder = ExecutionRecorder(run_id="fault-test")

        ctx = InterceptionContext(
            agent_id="reasoning",
            messages=[HumanMessage(content="Input")],
            step_index=1,
        )
        ctx.fault_applied = "context_truncation"
        ctx = recorder.before_call(ctx)
        ctx.response = AIMessage(content="Truncated output")
        recorder.after_call(ctx)

        assert recorder.recording.steps[0].fault_applied == "context_truncation"

    def test_agentstress_recorder_save_load(self, tmp_path):
        recorder = ExecutionRecorder(run_id="save-test", master_seed=42)

        ctx = InterceptionContext(
            agent_id="triage",
            messages=[HumanMessage(content="Patient data")],
            step_index=1,
        )
        ctx = recorder.before_call(ctx)
        ctx.response = AIMessage(content="ESI Level 2")
        recorder.after_call(ctx)

        path = recorder.save(tmp_path / "recording.json")
        assert path.exists()

        loaded = ExecutionRecorder.load(path)
        assert loaded.run_id == "save-test"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].output_content == "ESI Level 2"

    def test_agentstress_recorder_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            ExecutionRecorder.load("/nonexistent/recording.json")

    def test_agentstress_recorder_finalize(self):
        recorder = ExecutionRecorder(run_id="final")
        recording = recorder.finalize()
        assert recording.end_time > 0
        assert recording.duration_s >= 0

    def test_agentstress_recorder_reset(self):
        recorder = ExecutionRecorder(run_id="reset-test")
        ctx = InterceptionContext(
            agent_id="a", messages=[HumanMessage(content="X")], step_index=1,
        )
        ctx = recorder.before_call(ctx)
        ctx.response = AIMessage(content="Y")
        recorder.after_call(ctx)

        assert len(recorder.recording.steps) == 1
        recorder.reset()
        assert len(recorder.recording.steps) == 0

    def test_agentstress_recorder_name(self):
        recorder = ExecutionRecorder()
        assert recorder.name == "execution-recorder"


# --- ReplayPlayer ---

class TestAgentStressReplayPlayer:

    @pytest.fixture
    def agentstress_recording(self):
        rec = Recording(run_id="replay-test", master_seed=42)
        for i, agent_id in enumerate(["triage", "reasoning", "report"]):
            rec.steps.append(RecordedStep(
                agent_id=agent_id,
                step_index=i,
                input_messages=[{"role": "human", "content": f"Input {i}"}],
                output_content=f"Output from {agent_id}",
                fault_applied="context_truncation" if agent_id == "reasoning" else None,
                random_seed=42 + i,
            ))
        return rec

    def test_agentstress_player_init(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        assert player.total_steps == 3
        assert player.cursor == 0
        assert not player.is_complete

    def test_agentstress_player_before_after_call(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)

        ctx = InterceptionContext(
            agent_id="triage",
            messages=[HumanMessage(content="Test")],
            step_index=0,
        )
        ctx = player.before_call(ctx)
        assert ctx.metadata["replay_mode"]

        ctx.response = AIMessage(content="Will be replaced")
        ctx = player.after_call(ctx)

        assert ctx.response.content == "Output from triage"
        assert player.cursor == 1

    def test_agentstress_player_full_playback(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)

        for i in range(3):
            ctx = InterceptionContext(
                agent_id=agentstress_recording.steps[i].agent_id,
                messages=[HumanMessage(content="Input")],
                step_index=i,
            )
            ctx = player.before_call(ctx)
            ctx = player.after_call(ctx)

        assert player.is_complete
        assert len(player.playback_log) == 3

    def test_agentstress_player_navigation(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)

        step = player.step_forward()
        assert step is not None
        assert player.cursor == 1

        step = player.step_backward()
        assert step is not None
        assert player.cursor == 0

        step = player.jump_to(2)
        assert step is not None
        assert step.agent_id == "report"

        assert player.jump_to(99) is None

    def test_agentstress_player_rewind(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        player.jump_to(2)
        player.rewind()
        assert player.cursor == 0
        assert len(player.playback_log) == 0

    def test_agentstress_player_get_agent_steps(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        triage_steps = player.get_agent_steps("triage")
        assert len(triage_steps) == 1

    def test_agentstress_player_get_faulted_steps(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        faulted = player.get_faulted_steps()
        assert len(faulted) == 1
        assert faulted[0].agent_id == "reasoning"

    def test_agentstress_player_diff_outputs(self, agentstress_recording):
        # Create a second recording with one different output
        rec2 = Recording(run_id="diff-test", master_seed=42)
        for i, agent_id in enumerate(["triage", "reasoning", "report"]):
            content = "DIFFERENT output" if agent_id == "reasoning" else f"Output from {agent_id}"
            rec2.steps.append(RecordedStep(
                agent_id=agent_id, step_index=i,
                input_messages=[], output_content=content,
            ))

        player = ReplayPlayer(agentstress_recording)
        diffs = player.diff_outputs(rec2)
        assert len(diffs) == 1
        assert diffs[0]["agent_id"] == "reasoning"
        assert diffs[0]["status"] == "different"

    def test_agentstress_player_summary(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        s = player.summary()
        assert s["total_steps"] == 3
        assert len(s["agents"]) == 3
        assert s["faulted_steps"] == 1
        assert s["playback_progress"] == "0/3"

    def test_agentstress_player_name(self, agentstress_recording):
        player = ReplayPlayer(agentstress_recording)
        assert player.name == "replay-player"
