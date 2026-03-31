"""Replay player: deterministic playback of recorded stress test runs.

Loads a Recording and replays it step by step, returning the exact same
LLM responses without making actual API calls. Supports time-travel
navigation through the replayed execution.
"""

from __future__ import annotations

import random
from typing import Any

from langchain_core.messages import AIMessage

from proxy.agentstress_proxy_intercept import InterceptionContext
from replay.agentstress_replay_record import Recording, RecordedStep


class ReplayPlayer:
    """Replays a recorded execution, injecting recorded responses instead of calling LLMs.

    Acts as an Interceptor in the pipeline. On before_call, restores the
    random seed; on after_call, replaces the LLM response with the recorded one.
    Also provides time-travel navigation through the recording.
    """

    def __init__(self, recording: Recording) -> None:
        self._recording = recording
        self._steps = list(recording.steps)
        self._cursor = 0
        self._playback_log: list[RecordedStep] = []

    @property
    def name(self) -> str:
        return "replay-player"

    @property
    def recording(self) -> Recording:
        return self._recording

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    @property
    def is_complete(self) -> bool:
        return self._cursor >= len(self._steps)

    @property
    def current_step(self) -> RecordedStep | None:
        if 0 <= self._cursor < len(self._steps):
            return self._steps[self._cursor]
        return None

    @property
    def playback_log(self) -> list[RecordedStep]:
        return list(self._playback_log)

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        """Restore random seed from recording for deterministic fault behavior."""
        if self._cursor < len(self._steps):
            step = self._steps[self._cursor]
            random.seed(step.random_seed)
            ctx.metadata["replay_mode"] = True
            ctx.metadata["replay_step"] = self._cursor
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        """Replace LLM response with the recorded output."""
        if self._cursor < len(self._steps):
            step = self._steps[self._cursor]
            ctx.response = AIMessage(content=step.output_content)
            ctx.fault_applied = step.fault_applied
            self._playback_log.append(step)
            self._cursor += 1
        return ctx

    # --- Time-travel navigation ---

    def step_forward(self) -> RecordedStep | None:
        """Advance cursor by one step without executing."""
        if self._cursor < len(self._steps) - 1:
            self._cursor += 1
            return self._steps[self._cursor]
        return None

    def step_backward(self) -> RecordedStep | None:
        """Move cursor back one step."""
        if self._cursor > 0:
            self._cursor -= 1
            return self._steps[self._cursor]
        return None

    def jump_to(self, index: int) -> RecordedStep | None:
        """Jump to a specific step index."""
        if 0 <= index < len(self._steps):
            self._cursor = index
            return self._steps[self._cursor]
        return None

    def rewind(self) -> None:
        """Reset cursor to the beginning."""
        self._cursor = 0
        self._playback_log.clear()

    # --- Inspection ---

    def get_step(self, index: int) -> RecordedStep | None:
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def get_agent_steps(self, agent_id: str) -> list[RecordedStep]:
        return [s for s in self._steps if s.agent_id == agent_id]

    def get_faulted_steps(self) -> list[RecordedStep]:
        return [s for s in self._steps if s.fault_applied is not None]

    def diff_outputs(self, other_recording: Recording) -> list[dict[str, Any]]:
        """Compare this recording's outputs against another recording.

        Returns a list of diffs for steps where outputs differ.
        """
        diffs = []
        max_len = max(len(self._steps), len(other_recording.steps))
        for i in range(max_len):
            a = self._steps[i] if i < len(self._steps) else None
            b = other_recording.steps[i] if i < len(other_recording.steps) else None

            if a is None or b is None:
                diffs.append({
                    "step_index": i,
                    "status": "missing",
                    "this": a.output_content if a else None,
                    "other": b.output_content if b else None,
                })
            elif a.output_content != b.output_content:
                diffs.append({
                    "step_index": i,
                    "agent_id": a.agent_id,
                    "status": "different",
                    "this": a.output_content[:200],
                    "other": b.output_content[:200],
                })

        return diffs

    def summary(self) -> dict[str, Any]:
        agents = set(s.agent_id for s in self._steps)
        faulted = sum(1 for s in self._steps if s.fault_applied)
        return {
            "run_id": self._recording.run_id,
            "total_steps": len(self._steps),
            "cursor": self._cursor,
            "is_complete": self.is_complete,
            "agents": sorted(agents),
            "faulted_steps": faulted,
            "playback_progress": f"{self._cursor}/{len(self._steps)}",
        }
