"""Phase 1 exit criterion: a drag-free trajectory must match the closed form.

Under gravity alone the equation of motion is ``p(t) = p0 + v0 t + g t^2 / 2``,
a quadratic. RK4 integrates polynomials up to fourth order exactly, so the only
error here is floating-point accumulation over ten thousand steps — which lands
around 1e-11, five orders of magnitude inside the 1e-6 the criterion asks for.

That headroom is the point. Any real defect in the integrator, the state
packing or the gravity vector shows up as an error many orders of magnitude
larger, so this test does not need a carefully-tuned tolerance to be sensitive.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.airframe.aero import Aerodynamics
from interceptor.core.state import EntityState, Vector
from interceptor.core.world import STANDARD_GRAVITY, World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import break_turn
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

DURATION = 10.0
DT = 1e-3
LAUNCH_SPEED = 300.0


def _drag_free_world() -> tuple[World, EntityState]:
    """A world with gravity only, holding one unpowered missile at 45 degrees."""
    component = LAUNCH_SPEED / np.sqrt(2.0)
    initial = EntityState(
        pos=np.zeros(3),
        vel=np.array([0.0, component, component]),
        mass=85.0,
    )
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", initial))
    return world, initial


def _analytic_position(initial: EntityState, t: Vector) -> Vector:
    """Closed-form position at each time in ``t``, shape ``(len(t), 3)``."""
    gravity = np.array([0.0, 0.0, -STANDARD_GRAVITY])
    t_col = t[:, np.newaxis]
    return np.asarray(
        initial.pos + initial.vel * t_col + 0.5 * gravity * t_col**2, dtype=np.float64
    )


def _range_at_ground_crossing(position: Vector) -> float:
    """Downrange distance where the trajectory crosses z = 0, by interpolation.

    Reading the last sample below ground would be wrong by up to half a
    timestep of travel — 0.1 m at 200 m/s, which swamps the thing being tested.
    Interpolating between the two samples that straddle the crossing removes
    that error entirely.

    This is a small rehearsal of the technique Phase 3 needs for miss distance,
    where the same sampling error is roughly a metre and would make one guidance
    law indistinguishable from another.
    """
    height = position[:, 2]
    below = np.flatnonzero(height < 0.0)
    if below.size == 0:
        msg = "trajectory never reached the ground"
        raise AssertionError(msg)

    i = int(below[0])
    fraction = height[i - 1] / (height[i - 1] - height[i])
    downrange = position[:, 1]
    return float(downrange[i - 1] + fraction * (downrange[i] - downrange[i - 1]))


def test_drag_free_trajectory_matches_the_analytic_parabola() -> None:
    world, initial = _drag_free_world()
    result = run(world, duration=DURATION, dt=DT, record_hz=1000.0)

    simulated = result.recorder.position("missile")
    expected = _analytic_position(initial, result.recorder.time)

    error = np.abs(simulated - expected).max()
    assert error < 1e-6, f"worst position error was {error:.3e} m"


def test_drag_free_velocity_matches_the_analytic_solution() -> None:
    world, initial = _drag_free_world()
    result = run(world, duration=DURATION, dt=DT, record_hz=1000.0)

    gravity = np.array([0.0, 0.0, -STANDARD_GRAVITY])
    expected = initial.vel + gravity * result.recorder.time[:, np.newaxis]

    error = np.abs(result.recorder.velocity("missile") - expected).max()
    assert error < 1e-6, f"worst velocity error was {error:.3e} m/s"


def test_horizontal_velocity_is_untouched_by_gravity() -> None:
    world, initial = _drag_free_world()
    result = run(world, duration=DURATION, dt=DT, record_hz=1000.0)

    horizontal = result.recorder.velocity("missile")[:, :2]
    assert np.allclose(horizontal, initial.vel[:2], atol=1e-9)


def test_an_unpowered_missile_holds_its_mass() -> None:
    world, initial = _drag_free_world()
    result = run(world, duration=DURATION, dt=DT)

    assert np.allclose(result.recorder.mass("missile"), initial.mass, atol=1e-12)


@pytest.mark.parametrize("elevation_deg", [15.0, 30.0, 45.0, 60.0, 80.0])
def test_range_and_apex_match_the_textbook_formulae(elevation_deg: float) -> None:
    """Independent check against the schoolbook results, not the same algebra.

    Range ``v^2 sin(2 theta) / g`` and apex ``v^2 sin^2(theta) / 2g`` are derived
    differently from the position equation above, so agreeing with both is a
    stronger statement than agreeing with either.
    """
    elevation = np.radians(elevation_deg)
    initial = EntityState(
        pos=np.zeros(3),
        vel=np.array([0.0, LAUNCH_SPEED * np.cos(elevation), LAUNCH_SPEED * np.sin(elevation)]),
        mass=85.0,
    )
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", initial))

    flight_time = 2.0 * LAUNCH_SPEED * np.sin(elevation) / STANDARD_GRAVITY
    result = run(world, duration=flight_time * 1.05, dt=DT, record_hz=1000.0)

    position = result.recorder.position("missile")
    expected_range = LAUNCH_SPEED**2 * np.sin(2.0 * elevation) / STANDARD_GRAVITY
    expected_apex = (LAUNCH_SPEED * np.sin(elevation)) ** 2 / (2.0 * STANDARD_GRAVITY)

    assert position[:, 2].max() == pytest.approx(expected_apex, abs=1e-3)
    assert _range_at_ground_crossing(position) == pytest.approx(expected_range, abs=0.01)


# --------------------------------------------------------------------------
# Induced drag
# --------------------------------------------------------------------------
def test_a_body_pulling_no_g_has_only_zero_lift_drag() -> None:
    """The guarantee that made the change safe: ballistics are untouched.

    Every result from Phase 1 onwards was computed without induced drag, so if
    a non-manoeuvring body saw any of it, the parabola test would have moved.
    """
    aero = Aerodynamics()
    assert aero.total_drag_coefficient(0.0, 500.0, 85.0, 1.0) == pytest.approx(
        aero.drag_coefficient
    )


def test_turning_costs_drag() -> None:
    """The whole point. Before this, the airframe manoeuvred for free."""
    aero = Aerodynamics()
    gentle = aero.total_drag_coefficient(2.0 * 9.80665, 546.0, 70.0, 1.089)
    hard = aero.total_drag_coefficient(10.0 * 9.80665, 546.0, 70.0, 1.089)
    assert hard > gentle > aero.drag_coefficient
    # Induced drag goes as the square of the normal force, so five times the g
    # is twenty-five times the induced term.
    induced_gentle = gentle - aero.drag_coefficient
    induced_hard = hard - aero.drag_coefficient
    assert induced_hard == pytest.approx(25.0 * induced_gentle, rel=1e-6)


def test_the_induced_drag_factor_follows_from_the_lift_curve() -> None:
    """``k = 1 / Cn_alpha``, and the slope is peak lift over the angle it needs.

    Asserted as a relationship rather than a magic number, so that changing the
    airframe's lift capability changes what its turns cost, automatically and in
    the right direction.
    """
    aero = Aerodynamics(max_lift_coefficient=2.5, peak_lift_angle_deg=25.0)
    assert aero.induced_drag_factor == pytest.approx(np.deg2rad(25.0) / 2.5)

    # A more efficient airframe — the same lift at a smaller angle — pays less.
    efficient = Aerodynamics(max_lift_coefficient=2.5, peak_lift_angle_deg=15.0)
    assert efficient.induced_drag_factor < aero.induced_drag_factor


def test_induced_drag_is_bounded_by_the_lift_limit() -> None:
    """It cannot run away, because the airframe cannot exceed its own max lift.

    Worth pinning down: an unbounded ``k * Cn^2`` looks alarming, but ``Cn`` is
    capped by ``available_lateral_acceleration``, so the worst case is a fixed
    multiple of the zero-lift drag rather than something that diverges.
    """
    aero = Aerodynamics()
    speed, mass, density = 546.0, 70.0, 1.089
    limit = aero.available_lateral_acceleration(speed, mass, density)
    worst = aero.total_drag_coefficient(limit, speed, mass, density)
    ceiling = aero.drag_coefficient + aero.induced_drag_factor * aero.max_lift_coefficient**2
    assert worst <= ceiling + 1e-9
    assert worst / aero.drag_coefficient < 6.0


def test_a_manoeuvring_missile_arrives_slower_than_a_straight_one() -> None:
    """End to end: the correction has to show up in the trajectory, not just
    in a coefficient."""
    straight = scenarios.head_on()
    world, detector = straight.build(ProportionalNavigation(3.0))
    calm = run(world, duration=straight.duration, dt=1e-3, stop=detector)

    hard = scenarios.crossing().with_manoeuvre(break_turn(7.0, start_time=6.0))
    world, detector = hard.build(ProportionalNavigation(3.0))
    working = run(world, duration=hard.duration, dt=1e-3, stop=detector)

    assert working.recorder.speed("missile")[-1] < calm.recorder.speed("missile")[-1]
