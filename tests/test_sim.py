"""Scheduler, recorder and the run loop."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.core.world import World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import (
    Target,
    _horizontal_right_of,
    _lift_axis_of,
    barrel_roll,
    break_turn,
    jink,
    straight_and_level,
    weave,
)
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


# --------------------------------------------------------------------------
# Three-dimensional manoeuvres
# --------------------------------------------------------------------------
FLYING_NORTH = EntityState(pos=np.zeros(3), vel=np.array([0.0, 250.0, 0.0]))


def test_the_manoeuvre_frame_is_right_handed() -> None:
    """Right is east of north, and lift is up. Get this wrong and every banked
    manoeuvre goes the wrong way."""
    right = _horizontal_right_of(FLYING_NORTH.vel)
    lift = _lift_axis_of(FLYING_NORTH.vel)
    assert right == pytest.approx([1.0, 0.0, 0.0])
    assert lift == pytest.approx([0.0, 0.0, 1.0])
    assert float(np.dot(right, lift)) == pytest.approx(0.0)


def test_a_flat_weave_stays_in_the_horizontal_plane() -> None:
    """The default must be exactly what it was before banking existed."""
    flat = weave(6.0, 4.0)
    for t in np.linspace(0.0, 4.0, 9):
        assert flat(float(t), FLYING_NORTH)[2] == pytest.approx(0.0, abs=1e-9)


def test_a_ninety_degree_bank_weaves_vertically() -> None:
    vertical = weave(6.0, 4.0, bank_deg=90.0)
    command = vertical(0.0, FLYING_NORTH)
    assert abs(command[2]) > 0.0
    assert command[0] == pytest.approx(0.0, abs=1e-9)


def test_a_barrel_roll_holds_its_magnitude_and_rotates_its_direction() -> None:
    """Constant g, turning direction — which is what makes it a helix rather
    than a weave, and what a plan view cannot distinguish."""
    roll = barrel_roll(5.0, 4.0)
    peak = 5.0 * 9.80665

    directions = []
    for t in (0.0, 1.0, 2.0, 3.0):
        command = roll(t, FLYING_NORTH)
        assert float(np.linalg.norm(command)) == pytest.approx(peak)
        directions.append(command / peak)

    # A quarter period apart, so successive directions are perpendicular.
    for earlier, later in pairwise(directions):
        assert float(np.dot(earlier, later)) == pytest.approx(0.0, abs=1e-9)
    # And half a period apart it has reversed.
    assert float(np.dot(directions[0], directions[2])) == pytest.approx(-1.0)


def test_a_barrel_roll_leaves_the_horizontal_plane() -> None:
    """The property the 3D viewer exists for."""
    roll = barrel_roll(5.0, 4.0)
    vertical = [abs(float(roll(float(t), FLYING_NORTH)[2])) for t in np.linspace(0, 4, 17)]
    assert max(vertical) > 0.9 * 5.0 * 9.80665


def test_a_jink_changes_direction_between_intervals_and_holds_within_one() -> None:
    evasion = jink(7.0, interval=1.5, seed=0)
    peak = 7.0 * 9.80665

    within = [evasion(t, FLYING_NORTH) for t in (0.1, 0.7, 1.4)]
    for command in within:
        assert command == pytest.approx(within[0])
        assert float(np.linalg.norm(command)) == pytest.approx(peak)

    assert evasion(1.6, FLYING_NORTH) != pytest.approx(within[0])


def test_a_jink_is_reproducible_from_its_seed() -> None:
    """Determinism from a seed, which is what makes a Monte Carlo run mean
    anything."""
    times = np.linspace(0.0, 12.0, 25)
    first = [jink(7.0, 1.5, seed=4)(float(t), FLYING_NORTH) for t in times]
    again = [jink(7.0, 1.5, seed=4)(float(t), FLYING_NORTH) for t in times]
    different = [jink(7.0, 1.5, seed=5)(float(t), FLYING_NORTH) for t in times]

    assert all(a == pytest.approx(b) for a, b in zip(first, again, strict=True))
    assert any(a != pytest.approx(b) for a, b in zip(first, different, strict=True))


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(weave, {"period": 0.0}), (barrel_roll, {"period": -1.0}), (jink, {"interval": 0.0})],
)
def test_nonsense_timings_are_rejected(factory: object, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        factory(6.0, **kwargs)  # type: ignore[operator]
