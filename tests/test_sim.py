"""Scheduler, recorder and the run loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.core.world import World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import Target, break_turn, straight_and_level, weave
from interceptor.sim.engagement import ground_impact, run
from interceptor.sim.recorder import Recorder
from interceptor.sim.scheduler import Scheduler


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------
def test_rates_divide_the_physics_step_exactly() -> None:
    scheduler = Scheduler(1000.0)
    assert scheduler.rate(100.0).interval_ticks == 10
    assert scheduler.rate(200.0).interval_ticks == 5
    assert scheduler.rate(1000.0).interval_ticks == 1


def test_a_rate_that_does_not_divide_evenly_is_rejected() -> None:
    """Silently rounding 30 Hz to 33 ticks would make the stated timing a lie."""
    with pytest.raises(ValueError, match="does not divide"):
        Scheduler(1000.0).rate(30.0)


def test_rate_fires_on_the_expected_ticks() -> None:
    rate = Scheduler(1000.0).rate(100.0)
    fired = [tick for tick in range(50) if rate.due(tick)]
    assert fired == [0, 10, 20, 30, 40]


def test_scheduler_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Scheduler(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        Scheduler(1000.0).rate(-5.0)


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------
def _one_entity_world() -> World:
    world = World()
    world.add(Missile("missile", EntityState(pos=np.zeros(3), vel=np.array([0.0, 100.0, 100.0]))))
    return world


def test_recorder_grows_beyond_its_initial_capacity() -> None:
    world = _one_entity_world()
    recorder = Recorder(world.names, capacity=2)
    for i in range(10):
        recorder.record(float(i), world)
    assert len(recorder) == 10
    assert recorder.time.shape == (10,)


def test_recorder_rejects_a_zero_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be at least 1"):
        Recorder(("missile",), capacity=0)


def test_recorder_round_trips_through_a_file(tmp_path: Path) -> None:
    world = _one_entity_world()
    result = run(world, duration=1.0, dt=1e-3, record_hz=100.0)
    path = tmp_path / "run.npz"
    result.recorder.save(path)

    with np.load(path) as loaded:
        assert np.array_equal(loaded["missile.pos"], result.recorder.position("missile"))
        assert np.array_equal(loaded["time"], result.recorder.time)


# --------------------------------------------------------------------------
# Run loop
# --------------------------------------------------------------------------
def test_run_samples_at_the_requested_rate() -> None:
    result = run(_one_entity_world(), duration=2.0, dt=1e-3, record_hz=100.0)
    assert len(result.recorder) == 201  # inclusive of both endpoints
    assert result.recorder.time[-1] == pytest.approx(2.0)


def test_run_reports_why_it_stopped() -> None:
    result = run(_one_entity_world(), duration=1.0, dt=1e-3)
    assert result.reason == "duration elapsed"


def test_ground_impact_terminates_the_run_early() -> None:
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", EntityState(pos=np.array([0.0, 0.0, 100.0]), vel=np.zeros(3))))
    result = run(world, duration=60.0, dt=1e-3, stop=ground_impact("missile"))

    assert "reached the ground" in result.reason
    # Free fall from 100 m takes sqrt(2h/g) ~ 4.515 s.
    assert result.end_time == pytest.approx(4.515, abs=0.01)


def test_time_does_not_drift_over_a_long_run() -> None:
    """Time is tick * dt, not an accumulated sum, so it stays exact."""
    result = run(_one_entity_world(), duration=100.0, dt=1e-3, record_hz=1.0)
    assert result.recorder.time[-1] == pytest.approx(100.0, abs=1e-12)


def test_run_rejects_a_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        run(_one_entity_world(), duration=0.0)


# --------------------------------------------------------------------------
# Target manoeuvres
# --------------------------------------------------------------------------
def test_a_straight_target_holds_its_velocity() -> None:
    initial = EntityState(pos=np.array([8000.0, 0.0, 3000.0]), vel=np.array([-250.0, 0.0, 0.0]))
    world = World()
    world.add(Target("target", initial, straight_and_level()))

    result = run(world, duration=10.0, dt=1e-3, record_hz=100.0)
    assert np.allclose(result.recorder.velocity("target"), initial.vel, atol=1e-9)
    assert result.recorder.position("target")[-1][0] == pytest.approx(5500.0, abs=1e-6)


def test_a_weaving_target_manoeuvres_without_gaining_or_losing_speed() -> None:
    """A weave turns the target without accelerating it along its own path.

    The commanded acceleration is always perpendicular to the velocity, so
    speed is a conserved quantity — a sharper invariant than any statement
    about where the target ends up, and one that catches a mis-built right-hand
    vector immediately.
    """
    initial = EntityState(pos=np.zeros(3), vel=np.array([0.0, 250.0, 0.0]))
    world = World()
    world.add(Target("target", initial, weave(amplitude_g=6.0, period=5.0)))

    result = run(world, duration=5.0, dt=1e-3, record_hz=100.0)
    speed = result.recorder.speed("target")
    lateral = result.recorder.position("target")[:, 0]

    assert np.abs(speed - 250.0).max() < 1e-6, "a perpendicular force changed the speed"
    assert np.abs(lateral).max() > 50.0, "the target did not actually manoeuvre"


def test_a_weave_reverses_direction_each_half_cycle() -> None:
    initial = EntityState(pos=np.zeros(3), vel=np.array([0.0, 250.0, 0.0]))
    world = World()
    world.add(Target("target", initial, weave(amplitude_g=6.0, period=4.0)))

    result = run(world, duration=8.0, dt=1e-3, record_hz=100.0)
    lateral_velocity = result.recorder.velocity("target")[:, 0]
    assert lateral_velocity.max() > 10.0
    assert lateral_velocity.min() < -10.0


def test_a_break_turn_only_starts_when_told() -> None:
    initial = EntityState(pos=np.zeros(3), vel=np.array([0.0, 250.0, 0.0]))
    world = World()
    world.add(Target("target", initial, break_turn(amplitude_g=7.0, start_time=3.0)))

    result = run(world, duration=6.0, dt=1e-3, record_hz=100.0)
    lateral = np.abs(result.recorder.position("target")[:, 0])
    times = result.recorder.time

    assert lateral[times <= 3.0].max() < 1e-9
    assert lateral[-1] > 100.0
