"""The run loop: step the world at a fixed rate until something ends it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from interceptor.core.world import World
from interceptor.sim.recorder import Recorder
from interceptor.sim.scheduler import Scheduler

__all__ = ["RunResult", "StopCondition", "ground_impact", "run"]

#: Returns a reason string to stop the run, or ``None`` to continue.
StopCondition = Callable[[float, World], "str | None"]


@dataclass(frozen=True)
class RunResult:
    """Everything a completed run produced."""

    recorder: Recorder
    reason: str
    end_time: float


def ground_impact(name: str, altitude: float = 0.0) -> StopCondition:
    """Stop when the named entity falls to or below ``altitude``."""

    def condition(t: float, world: World) -> str | None:
        del t
        if world[name].state.altitude <= altitude:
            return f"{name} reached the ground"
        return None

    return condition


def run(
    world: World,
    *,
    duration: float,
    dt: float = 1e-3,
    record_hz: float = 200.0,
    guidance_hz: float = 100.0,
    stop: StopCondition | None = None,
) -> RunResult:
    """Advance ``world`` for up to ``duration`` seconds.

    Args:
        world: The world to step. It is mutated in place.
        duration: Maximum simulated time, seconds.
        dt: Physics timestep. 1 ms is the project default.
        record_hz: Sampling rate for the recorder. Must divide ``1 / dt``.
        guidance_hz: Rate at which guided entities re-plan. Must divide
            ``1 / dt``. 100 Hz under a 1 kHz physics step means every guidance
            command is held for ten integration steps.
        stop: Optional early-termination condition, evaluated once per physics
            tick after guidance and recording, before the step.

    Each tick runs in a fixed order: guidance, then recording, then the stop
    check, then the integration step. Guidance comes first so that a recorded
    sample shows the command that is actually in force over the step that
    follows it, rather than the previous one.

    Time is computed as ``tick * dt`` rather than accumulated by repeated
    addition, so it cannot drift over a long run.
    """
    if duration <= 0.0:
        msg = f"duration must be positive, got {duration}"
        raise ValueError(msg)

    scheduler = Scheduler(1.0 / dt)
    record_rate = scheduler.rate(record_hz)
    guidance_rate = scheduler.rate(guidance_hz)
    guidance_dt = guidance_rate.interval_ticks * dt

    total_ticks = round(duration / dt)
    capacity = total_ticks // record_rate.interval_ticks + 2
    recorder = Recorder(world.names, capacity)

    reason = "duration elapsed"
    t = 0.0

    for tick in range(total_ticks + 1):
        t = tick * dt

        if guidance_rate.due(tick):
            for entity in world.entities:
                entity.update_guidance(t, guidance_dt)

        if record_rate.due(tick):
            recorder.record(t, world)

        if stop is not None:
            triggered = stop(t, world)
            if triggered is not None:
                reason = triggered
                break

        if tick < total_ticks:
            world.step(t, dt)

    return RunResult(recorder=recorder, reason=reason, end_time=t)
