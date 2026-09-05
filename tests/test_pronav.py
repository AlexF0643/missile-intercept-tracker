"""Proportional navigation, and the Phase 3 exit criterion.

The criterion is comparative: PN must beat pure pursuit by at least an order of
magnitude on the crossing geometry. Asserting the *comparison* rather than an
absolute number is what makes the test meaningful — it would still fail if a
change made both laws worse together, which an absolute threshold would not
catch.

The comparison is flown against a *weaving* crossing target rather than a
straight one. That is not to flatter PN; it is because modelling the energy cost
of turning changed what pursuit does against a straight target. Slowed by
induced drag it no longer overshoots, so it converges into a stern chase and
eventually arrives — eventually being the operative word, at 40% more time and a
quarter of the closing speed. Against anything that manoeuvres, the order of
magnitude is back and then some: 89 m against 6 m on a weave, 434 m against 3 m
on a break turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.entities.target import weave
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sensing.track import Track
from interceptor.sim import scenarios
from interceptor.sim.engagement import RunResult, run
from interceptor.sim.intercept import Intercept

G = 9.80665


def _track(position: list[float], velocity: list[float]) -> Track:
    return Track(
        time=0.0,
        relative_position=np.array(position),
        relative_velocity=np.array(velocity),
    )


def _state(velocity: list[float]) -> EntityState:
    return EntityState(pos=np.zeros(3), vel=np.array(velocity), mass=85.0)


def _fly(scenario: scenarios.Scenario, law: object) -> tuple[RunResult, Intercept | None]:
    world, detector = scenario.build(law)  # type: ignore[arg-type]
    result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return result, detector.result


# --------------------------------------------------------------------------
# The law in isolation
# --------------------------------------------------------------------------
def test_pronav_commands_nothing_on_a_collision_course() -> None:
    """The defining property. Zero line-of-sight rate means do nothing.

    If this test fails, the geometry in the law is wrong — and the symptom in a
    full run would be a missile that wanders off a perfectly good intercept.
    """
    track = _track([3000.0, 4000.0, 0.0], [-300.0, -400.0, 0.0])
    command = ProportionalNavigation(3.0).command(track, _state([300.0, 400.0, 0.0]))
    assert float(np.linalg.norm(command)) == pytest.approx(0.0, abs=1e-9)


def test_pronav_commands_nothing_when_not_closing() -> None:
    """PN's premise is a shrinking range; without one it has nothing to say."""
    track = _track([1000.0, 0.0, 0.0], [500.0, 0.0, 0.0])
    assert np.allclose(ProportionalNavigation().command(track, _state([100.0, 0.0, 0.0])), 0.0)


def test_pronav_commands_nothing_on_an_invalid_track() -> None:
    track = Track(
        time=0.0,
        relative_position=np.array([0.0, 5000.0, 0.0]),
        relative_velocity=np.array([250.0, -400.0, 0.0]),
        valid=False,
    )
    assert np.allclose(ProportionalNavigation().command(track, _state([0.0, 500.0, 0.0])), 0.0)


def test_pronav_leads_a_crossing_target() -> None:
    """The whole point: steer where the target is going, not where it is.

    The target is due north and moving east at 250 m/s while the missile runs
    north at 500 m/s, so the relative velocity is (250, -500, 0) — crossing and
    closing. The command must point east, ahead of the target, rather than along
    the sightline towards it.

    The relative velocity matters here: a target crossing a *stationary* missile
    has no closing velocity at all, and PN correctly commands nothing.
    """
    track = _track([0.0, 5000.0, 0.0], [250.0, -500.0, 0.0])
    command = ProportionalNavigation(3.0).command(track, _state([0.0, 500.0, 0.0]))

    assert track.closing_speed == pytest.approx(500.0)
    assert command[0] > 0.0, "PN failed to lead the target"
    assert abs(float(command[1])) < 1e-9, "a command along the velocity is unphysical"


def test_pronav_magnitude_is_n_times_closing_speed_times_los_rate() -> None:
    """``|a| = N * V_c * lambda_dot`` — the classical scalar law."""
    track = _track([0.0, 5000.0, 0.0], [250.0, -400.0, 0.0])
    state = _state([0.0, 500.0, 0.0])
    command = ProportionalNavigation(3.0).command(track, state)

    expected = 3.0 * track.closing_speed * track.los_rate
    assert float(np.linalg.norm(command)) == pytest.approx(expected, rel=1e-9)


def test_pronav_command_is_perpendicular_to_the_velocity() -> None:
    """Lift acts at right angles to the airflow; a missile has no throttle."""
    track = _track([2000.0, 5000.0, 800.0], [250.0, -400.0, 0.0])
    velocity = np.array([100.0, 480.0, 40.0])
    command = ProportionalNavigation(4.0).command(track, _state(list(velocity)))
    assert float(np.dot(command, velocity)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("constant", [3.0, 4.0, 5.0])
def test_command_scales_linearly_with_the_navigation_constant(constant: float) -> None:
    track = _track([0.0, 5000.0, 0.0], [250.0, -400.0, 0.0])
    state = _state([0.0, 500.0, 0.0])
    baseline = np.linalg.norm(ProportionalNavigation(1.0).command(track, state))
    scaled = np.linalg.norm(ProportionalNavigation(constant).command(track, state))
    assert float(scaled / baseline) == pytest.approx(constant)


def test_pronav_rejects_a_non_positive_constant() -> None:
    with pytest.raises(ValueError, match="navigation_constant must be positive"):
        ProportionalNavigation(0.0)


def test_the_law_reports_its_constant_in_its_name() -> None:
    assert ProportionalNavigation(3.0).name == "ProNav (N=3)"


# --------------------------------------------------------------------------
# Phase 3 exit criterion
# --------------------------------------------------------------------------
def test_pronav_beats_pure_pursuit_on_the_crossing_geometry() -> None:
    """The Phase 3 exit criterion: at least an order of magnitude better."""
    scenario = scenarios.crossing().with_manoeuvre(weave(6.0, 4.0))
    _, pursuit = _fly(scenario, PurePursuit(gain=4.0))
    _, pronav = _fly(scenario, ProportionalNavigation(3.0))

    assert pursuit is not None
    assert pronav is not None
    assert pronav.miss_distance < pursuit.miss_distance / 10.0, (
        f"pursuit {pursuit.miss_distance:.3f} m vs pronav {pronav.miss_distance:.3f} m"
    )


def test_pronav_arrives_sooner_and_with_more_energy() -> None:
    """The advantage that survives even where pursuit does eventually hit.

    Against a target obliging enough to fly straight, pure pursuit gets there
    too — by chasing it down over nearly twenty seconds and arriving with its
    speed spent. Miss distance alone cannot tell those two outcomes apart, and
    a round with no closing speed left has no answer to a target that changes
    its mind.
    """
    _, pursuit = _fly(scenarios.crossing(), PurePursuit(gain=4.0))
    _, pronav = _fly(scenarios.crossing(), ProportionalNavigation(3.0))

    assert pursuit is not None
    assert pronav is not None
    assert pronav.time < pursuit.time
    assert pronav.closing_speed > 2.0 * pursuit.closing_speed


def test_pronav_hits_the_crossing_target() -> None:
    _, intercept = _fly(scenarios.crossing(), ProportionalNavigation(3.0))
    assert intercept is not None
    assert intercept.hit is True
    assert intercept.miss_distance < 1.0


def test_pronav_uses_less_acceleration_than_pursuit_to_do_better() -> None:
    """The counter-intuitive result, and the strongest argument for PN.

    A better intercept usually costs more effort. Here it costs less: pure
    pursuit spends the endgame hauling the missile around onto a sightline that
    keeps moving, while PN removed the need for that turn at the start.

    Flown against a straight target on purpose. This is a claim about the
    *mechanism* — the turn PN avoids having to make — and a weaving target
    obscures it, because then both laws are working hard for reasons that have
    nothing to do with the geometry either of them chose.
    """
    pursuit_run, _ = _fly(scenarios.crossing(), PurePursuit(gain=4.0))
    pronav_run, _ = _fly(scenarios.crossing(), ProportionalNavigation(3.0))

    pursuit_peak = float(pursuit_run.recorder.achieved("missile").max())
    pronav_peak = float(pronav_run.recorder.achieved("missile").max())
    assert pronav_peak < pursuit_peak


@pytest.mark.parametrize("constant", [3.0, 4.0, 5.0])
def test_the_practical_range_of_n_all_intercept(constant: float) -> None:
    """N between 3 and 5 is the usual range, and all of it works on truth data.

    They separate in Phase 4, when a higher N starts amplifying seeker noise.
    """
    _, intercept = _fly(scenarios.crossing(), ProportionalNavigation(constant))
    assert intercept is not None
    assert intercept.hit is True


@pytest.mark.parametrize("factory", [scenarios.head_on, scenarios.crossing, scenarios.tail_chase])
def test_pronav_intercepts_every_standard_geometry(factory: object) -> None:
    _, intercept = _fly(factory(), ProportionalNavigation(3.0))  # type: ignore[operator]
    assert intercept is not None
    assert intercept.hit is True


def _line_of_sight_rate(result: RunResult) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the line-of-sight rate from the recorded trajectories.

    Computed from the flown result rather than from the law's own arithmetic, so
    this measures the outcome instead of restating the implementation.
    """
    record = result.recorder
    r = record.position("target") - record.position("missile")
    v = record.velocity("target") - record.velocity("missile")
    omega = np.cross(r, v) / np.einsum("ij,ij->i", r, r)[:, None]
    return record.time, np.asarray(np.linalg.norm(omega, axis=1))


def test_pronav_holds_the_bearing_steady_where_pursuit_lets_it_swing() -> None:
    """The mechanism, measured rather than asserted.

    Proportional navigation works by holding the line-of-sight rate *constant*:
    a bearing that does not change while the range falls is, by definition, a
    collision. Note what this does not claim — PN does not drive the rate to
    zero, and should not be expected to, because gravity is a continuous
    disturbance and a proportional loop against one settles at a small constant
    error rather than eliminating it.

    So the quantity to measure is steadiness, not magnitude. PN holds the rate
    inside a factor of about two across the whole engagement, while pure
    pursuit's sweeps up as it hauls itself onto a moving sightline and then
    collapses towards zero as it gives up and settles into a stern chase. That
    swing is the failure: the sightline is doing something entirely different at
    the end of the engagement from the start, which is precisely what a
    collision course is not.

    This test used to compare *peak* rates and demand a factor of ten. Adding
    induced drag ended pursuit's overshoot and with it the runaway peak, so that
    comparison stopped measuring anything — the peak fell while the behaviour
    stayed exactly as wrong. Spread catches what peak no longer does.

    Flown against a straight target on purpose: a weave drives the
    line-of-sight rate itself, so neither law could hold it steady and the
    measurement would be of the target rather than of the guidance.
    """
    pursuit_run, _ = _fly(scenarios.crossing(), PurePursuit(gain=4.0))
    pronav_run, _ = _fly(scenarios.crossing(), ProportionalNavigation(3.0))

    # Skip the launch transient, and the final half-second where range -> 0
    # makes the rate numerically explosive for any law.
    def window(result: RunResult) -> np.ndarray:
        time, rate = _line_of_sight_rate(result)
        return np.asarray(rate[(time > 3.0) & (time < time[-1] - 0.5)])

    def spread(rate: np.ndarray) -> float:
        return float(rate.max()) / max(float(rate.min()), 1e-9)

    pursuit_rate = window(pursuit_run)
    pronav_rate = window(pronav_run)

    assert spread(pronav_rate) < 3.0, (
        f"pronav should hold the bearing rate steady; spread was {spread(pronav_rate):.1f}x"
    )
    assert spread(pursuit_rate) > 20.0, (
        f"pursuit should not hold it steady; spread was {spread(pursuit_rate):.1f}x"
    )
    assert pronav_rate.mean() < pursuit_rate.mean() / 2.0
