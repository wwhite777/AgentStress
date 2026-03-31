"""Fault scheduling: controls when and how faults are activated over time.

Wraps any BaseFaultInjector with temporal scheduling logic:
- continuous: fault active at every step
- burst: fault active for N consecutive steps, then inactive for M steps
- progressive: fault probability increases linearly over steps
- once: fault triggers exactly once, then deactivates
"""

from __future__ import annotations

from faults.agentstress_fault_base import BaseFaultInjector, FaultConfig, FaultSchedule
from proxy.agentstress_proxy_intercept import InterceptionContext


class ScheduledFaultWrapper:
    """Wraps a BaseFaultInjector with temporal scheduling logic.

    Acts as an Interceptor in the pipeline, delegating to the wrapped
    fault only when the schedule permits.
    """

    def __init__(self, fault: BaseFaultInjector) -> None:
        self.fault = fault
        self.schedule = fault.config.schedule
        self._step_count = 0
        self._triggered = False

        # Burst params
        self._burst_on = fault.config.params.get("burst_on_steps", 3)
        self._burst_off = fault.config.params.get("burst_off_steps", 5)

        # Progressive params
        self._progressive_start = fault.config.params.get("progressive_start", 0.0)
        self._progressive_end = fault.config.params.get("progressive_end", 1.0)
        self._progressive_ramp_steps = fault.config.params.get("progressive_ramp_steps", 10)

    @property
    def name(self) -> str:
        return f"scheduled-{self.fault.name}"

    def _is_active(self) -> bool:
        if self.schedule == FaultSchedule.CONTINUOUS:
            return True

        if self.schedule == FaultSchedule.ONCE:
            return not self._triggered

        if self.schedule == FaultSchedule.BURST:
            cycle_length = self._burst_on + self._burst_off
            position = (self._step_count - 1) % cycle_length
            return position < self._burst_on

        if self.schedule == FaultSchedule.PROGRESSIVE:
            return True  # always active, but probability varies

        return True

    def _get_effective_probability(self) -> float:
        if self.schedule != FaultSchedule.PROGRESSIVE:
            return self.fault.config.probability

        if self._progressive_ramp_steps <= 0:
            return self._progressive_end

        progress = min(1.0, self._step_count / self._progressive_ramp_steps)
        return self._progressive_start + (self._progressive_end - self._progressive_start) * progress

    def before_call(self, ctx: InterceptionContext) -> InterceptionContext:
        self._step_count += 1

        if not self._is_active():
            ctx.metadata["schedule_skipped"] = True
            ctx.metadata["schedule_type"] = self.schedule.value
            return ctx

        original_prob = self.fault.config.probability
        if self.schedule == FaultSchedule.PROGRESSIVE:
            self.fault.config.probability = self._get_effective_probability()
            ctx.metadata["schedule_effective_probability"] = self.fault.config.probability

        ctx = self.fault.before_call(ctx)

        if self.schedule == FaultSchedule.PROGRESSIVE:
            self.fault.config.probability = original_prob

        if ctx.fault_applied and self.schedule == FaultSchedule.ONCE:
            self._triggered = True

        ctx.metadata["schedule_step"] = self._step_count
        ctx.metadata["schedule_type"] = self.schedule.value
        return ctx

    def after_call(self, ctx: InterceptionContext) -> InterceptionContext:
        return self.fault.after_call(ctx)

    def reset(self) -> None:
        self.fault.reset()
        self._step_count = 0
        self._triggered = False

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def triggered(self) -> bool:
        return self._triggered


def wrap_with_schedule(fault: BaseFaultInjector) -> ScheduledFaultWrapper:
    """Convenience factory: wrap a fault injector with its configured schedule."""
    return ScheduledFaultWrapper(fault)
