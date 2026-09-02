"""The guidance interface, the track, the autopilot and pure pursuit."""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.airframe.autopilot import Autopilot, clamp_magnitude
from interceptor.core.state import EntityState, Vector
from interceptor.core.world import World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import Target
from interceptor.guidance.base import GuidanceLaw, perpendicular_component
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sensing.track import Track, TruthTrack
from interceptor.sim.engagement import run

G = 9.80665


# --------------------------------------------------------------------------
# Track geometry
# --------------------------------------------------------------------------
def _track(relative_position: list[float], relative_velocity: list[float]) -> Track:
    return Track(
        time=0.0,
        relative_position=np.array(relative_position),
        relative_velocity=np.array(relative_velocity),
    )


def test_range_and_line_of_sight() -> None:
    track = _track([3000.0, 4000.0, 0.0], [0.0, 0.0, 0.0])
    assert track.range == pytest.approx(5000.0)
    assert np.allclose(track.line_of_sight, [0.6, 0.8, 0.0])


def test_closing_speed_is_positive_while_closing() -> None:
    approaching = _track([1000.0, 0.0, 0.0], [-500.0, 0.0, 0.0])
    receding = _track([1000.0, 0.0, 0.0], [500.0, 0.0, 0.0])
    assert approaching.closing_speed == pytest.approx(500.0)
    assert receding.closing_speed == pytest.approx(-500.0)


def test_a_collision_course_has_zero_line_of_sight_rate() -> None:
    """Constant bearing, decreasing range. The condition PN exists to create.

    The relative velocity points exactly along the sightline, so the sightline
    translates without rotating and the rate is identically zero.
    """
    track = _track([3000.0, 4000.0, 0.0], [-300.0, -400.0, 0.0])
    assert track.los_rate == pytest.approx(0.0, abs=1e-15)
    assert track.closing_speed == pytest.approx(500.0)


def test_a_crossing_target_has_a_nonzero_line_of_sight_rate() -> None:
    track = _track([0.0, 5000.0, 0.0], [250.0, 0.0, 0.0])
    assert track.los_rate == pytest.approx(250.0 / 5000.0)


def test_time_to_go_is_infinite_when_not_closing() -> None:
    assert _track([1000.0, 0.0, 0.0], [500.0, 0.0, 0.0]).time_to_go == float("inf")
    assert _track([1000.0, 0.0, 0.0], [-500.0, 0.0, 0.0]).time_to_go == pytest.approx(2.0)


def test_truth_track_follows_the_target_as_it_moves() -> None:
    """The track holds the entity, not a snapshot — a frozen target is a real bug."""
    target = Target(
        "target", EntityState(pos=np.array([0.0, 1000.0, 0.0]), vel=np.array([0.0, -200.0, 0.0]))
    )
    source = TruthTrack(target)
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 100.0, 0.0]))

    first = source.update(0.0, missile)
    target.state = EntityState(pos=np.array([0.0, 800.0, 0.0]), vel=target.state.vel)
    second = source.update(1.0, missile)

    assert first.range == pytest.approx(1000.0)
    assert second.range == pytest.approx(800.0)


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------
def test_perpendicular_component_removes_the_along_track_part() -> None:
    reference = np.array([10.0, 0.0, 0.0])
    result = perpendicular_component(np.array([5.0, 3.0, 0.0]), reference)
    assert np.allclose(result, [0.0, 3.0, 0.0])
    assert float(np.dot(result, reference)) == pytest.approx(0.0)


def test_clamp_leaves_short_vectors_alone() -> None:
    vector = np.array([3.0, 4.0, 0.0])
    assert np.allclose(clamp_magnitude(vector, 10.0), vector)


def test_clamp_preserves_direction() -> None:
    clamped = clamp_magnitude(np.array([30.0, 40.0, 0.0]), 10.0)
    assert float(np.linalg.norm(clamped)) == pytest.approx(10.0)
    assert np.allclose(clamped / 10.0, [0.6, 0.8, 0.0])


# --------------------------------------------------------------------------
# Autopilot
# --------------------------------------------------------------------------
def test_autopilot_reaches_63_percent_after_one_time_constant() -> None:
    """The defining property of a first-order lag."""
    autopilot = Autopilot(time_constant=0.2)
    command = np.array([100.0, 0.0, 0.0])
    dt = 1e-3
    for _ in range(200):  # 0.2 s
        autopilot.update(command, dt, limit=1000.0)
    assert float(autopilot.achieved[0]) == pytest.approx(63.0, abs=1.5)


def test_a_zero_time_constant_responds_instantly() -> None:
    autopilot = Autopilot(time_constant=0.0)
    achieved = autopilot.update(np.array([50.0, 0.0, 0.0]), 0.01, limit=1000.0)
    assert np.allclose(achieved, [50.0, 0.0, 0.0])


def test_the_autopilot_never_exceeds_the_airframe_limit() -> None:
    autopilot = Autopilot(time_constant=0.0)
    achieved = autopilot.update(np.array([0.0, 500.0, 0.0]), 0.01, limit=100.0)
    assert float(np.linalg.norm(achieved)) == pytest.approx(100.0)


def test_autopilot_rejects_a_negative_time_constant() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        Autopilot(time_constant=-0.1)


# --------------------------------------------------------------------------
# Pure pursuit
# --------------------------------------------------------------------------
def _missile_state(velocity: list[float]) -> EntityState:
    return EntityState(pos=np.zeros(3), vel=np.array(velocity), mass=85.0)


def test_pursuit_commands_nothing_on_an_invalid_track() -> None:
    law = PurePursuit()
    track = Track(
        time=0.0,
        relative_position=np.array([0.0, 5000.0, 0.0]),
        relative_velocity=np.zeros(3),
        valid=False,
    )
    assert np.allclose(law.command(track, _missile_state([0.0, 500.0, 0.0])), 0.0)


def test_pursuit_commands_nothing_when_already_aimed_at_the_target() -> None:
    law = PurePursuit()
    track = _track([0.0, 5000.0, 0.0], [0.0, 0.0, 0.0])
    assert np.allclose(law.command(track, _missile_state([0.0, 500.0, 0.0])), 0.0, atol=1e-9)


def test_pursuit_commands_only_lateral_acceleration() -> None:
    """A missile has no throttle, so a command along the velocity is meaningless."""
    law = PurePursuit()
    track = _track([3000.0, 4000.0, 0.0], [0.0, 0.0, 0.0])
    velocity = np.array([0.0, 500.0, 0.0])
    command = law.command(track, _missile_state(list(velocity)))
    assert float(np.dot(command, velocity)) == pytest.approx(0.0, abs=1e-9)


def test_pursuit_turns_towards_the_target() -> None:
    law = PurePursuit()
    track = _track([3000.0, 4000.0, 0.0], [0.0, 0.0, 0.0])
    command = law.command(track, _missile_state([0.0, 500.0, 0.0]))
    # The target is off to the east, so the command must have an easterly part.
    assert command[0] > 0.0


def test_pursuit_command_scales_with_gain() -> None:
    track = _track([3000.0, 4000.0, 0.0], [0.0, 0.0, 0.0])
    state = _missile_state([0.0, 500.0, 0.0])
    weak = np.linalg.norm(PurePursuit(gain=2.0).command(track, state))
    strong = np.linalg.norm(PurePursuit(gain=4.0).command(track, state))
    assert float(strong / weak) == pytest.approx(2.0)


def test_pursuit_rejects_a_non_positive_gain() -> None:
    with pytest.raises(ValueError, match="gain must be positive"):
        PurePursuit(gain=0.0)


# --------------------------------------------------------------------------
# Guidance in the loop
# --------------------------------------------------------------------------
class _CountingLaw(GuidanceLaw):
    """Records how often it was asked, and what it was asked about."""

    def __init__(self) -> None:
        self.calls = 0
        self.times: list[float] = []

    def command(self, track: Track, missile: EntityState) -> Vector:
        del missile
        self.calls += 1
        self.times.append(track.time)
        return np.zeros(3, dtype=np.float64)


def _world_with(law: GuidanceLaw) -> World:
    world = World(WorldConfig(enable_gravity=False, enable_drag=False))
    target = Target(
        "target", EntityState(pos=np.array([0.0, 5000.0, 0.0]), vel=np.array([200.0, 0.0, 0.0]))
    )
    world.add(target)
    world.add(
        Missile(
            "missile",
            _missile_state([0.0, 500.0, 0.0]),
            guidance=law,
            track_source=TruthTrack(target),
            autopilot=Autopilot(0.0),
        )
    )
    return world


def test_guidance_runs_at_the_requested_rate() -> None:
    law = _CountingLaw()
    run(_world_with(law), duration=1.0, dt=1e-3, guidance_hz=100.0)
    assert law.calls == 101  # 0.00 s to 1.00 s inclusive


def test_guidance_rate_is_independent_of_the_physics_rate() -> None:
    law = _CountingLaw()
    run(_world_with(law), duration=1.0, dt=1e-3, guidance_hz=50.0)
    assert law.calls == 51


def test_a_guidance_rate_that_does_not_divide_evenly_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        run(_world_with(_CountingLaw()), duration=1.0, dt=1e-3, guidance_hz=30.0)


def test_an_unguided_missile_needs_no_track_source() -> None:
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", _missile_state([0.0, 500.0, 100.0])))
    result = run(world, duration=2.0, dt=1e-3)
    assert result.recorder.commanded("missile").max() == 0.0
