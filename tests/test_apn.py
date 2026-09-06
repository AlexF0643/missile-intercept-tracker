"""Augmented proportional navigation, and the boundary of its assumption.

APN adds a term for the target's own acceleration:

    a = N * V_c * (Omega x r_hat)  +  (N / 2) * a_t_perp

The coefficient is not tuning — it is the optimal-control solution for a
*constant-acceleration* target, under the same criterion that gives N = 3 for a
non-manoeuvring one.

The obvious thing to test is therefore what happens when the target's
acceleration is not constant, and the first version of this file tested exactly
that and drew the wrong conclusion from it. The lead term is a request for
additional lift, and what decides whether it helps is not how well the target
obliges the assumption but whether the airframe can meet the request. Both
halves are pinned below, the correction included, because it is the more useful
of the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.config import loads
from interceptor.core.state import EntityState
from interceptor.entities.missile import Missile
from interceptor.entities.target import (
    barrel_roll,
    break_turn,
    jink,
    straight_and_level,
    weave,
)
from interceptor.guidance.pronav import AugmentedProportionalNavigation, ProportionalNavigation
from interceptor.sensing.filters import ExtendedKalman
from interceptor.sensing.seeker import SeekerConfig
from interceptor.sensing.track import Track
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

G = 9.80665


def _fly(manoeuvre: object, law: object, *, seeker: bool = False, seed: int = 0) -> float | None:
    scenario = scenarios.crossing().with_manoeuvre(manoeuvre)  # type: ignore[arg-type]
    world, detector = scenario.build(
        law,  # type: ignore[arg-type]
        seeker=SeekerConfig() if seeker else None,
        estimator=ExtendedKalman() if seeker else None,
        seed=seed,
    )
    run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return None if detector.result is None else detector.result.miss_distance


# --------------------------------------------------------------------------
# The term itself
# --------------------------------------------------------------------------
def _closing_track(target_acceleration: np.ndarray | None) -> Track:
    """A track closing head-on along +north, so the sightline is +north."""
    return Track(
        time=1.0,
        relative_position=np.array([0.0, 2000.0, 0.0]),
        relative_velocity=np.array([0.0, -600.0, 0.0]),
        target_acceleration=target_acceleration,
    )


def test_with_no_target_acceleration_it_is_plain_pronav() -> None:
    """A track that cannot supply one must not be punished for it.

    An alpha-beta filter reports zero and a bare seeker track reports ``None``.
    Either way the law should quietly fall back rather than fail.
    """
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 600.0, 0.0]))
    plain = ProportionalNavigation(3.0)
    augmented = AugmentedProportionalNavigation(3.0)

    for absent in (None, np.zeros(3)):
        track = _closing_track(absent)
        assert augmented.command(track, missile) == pytest.approx(plain.command(track, missile))


def test_the_lead_term_is_half_the_navigation_constant_times_the_acceleration() -> None:
    """``N / 2`` exactly, and only the part across the sightline."""
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 600.0, 0.0]))
    across = np.array([40.0, 0.0, 0.0])

    plain = ProportionalNavigation(3.0).command(_closing_track(None), missile)
    augmented = AugmentedProportionalNavigation(3.0).command(_closing_track(across), missile)

    assert augmented - plain == pytest.approx(0.5 * 3.0 * across)


def test_acceleration_along_the_sightline_is_ignored() -> None:
    """It changes the closing speed, not the bearing, and this law steers on bearing."""
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 600.0, 0.0]))
    along = np.array([0.0, 60.0, 0.0])  # the sightline is +north

    plain = ProportionalNavigation(3.0).command(_closing_track(None), missile)
    augmented = AugmentedProportionalNavigation(3.0).command(_closing_track(along), missile)

    assert augmented == pytest.approx(plain)


def test_a_track_that_is_not_closing_gets_no_lead_either() -> None:
    """PN returns zero when not closing; adding a lead to that would be worse."""
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 600.0, 0.0]))
    opening = Track(
        time=1.0,
        relative_position=np.array([0.0, 2000.0, 0.0]),
        relative_velocity=np.array([0.0, 300.0, 0.0]),
        target_acceleration=np.array([40.0, 0.0, 0.0]),
    )
    assert AugmentedProportionalNavigation(3.0).command(opening, missile) == pytest.approx(
        np.zeros(3)
    )


def test_an_invalid_track_commands_nothing() -> None:
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 600.0, 0.0]))
    dropped = Track(
        time=1.0,
        relative_position=np.zeros(3),
        relative_velocity=np.zeros(3),
        target_acceleration=np.array([40.0, 0.0, 0.0]),
        valid=False,
    )
    assert AugmentedProportionalNavigation(3.0).command(dropped, missile) == pytest.approx(
        np.zeros(3)
    )


# --------------------------------------------------------------------------
# Where the assumption holds
# --------------------------------------------------------------------------
def test_a_truth_track_supplies_the_real_target_acceleration() -> None:
    """Without this, APN on perfect information would silently be plain PN.

    Which would make the perfect-information baseline useless for the one
    question it exists to answer: how well *could* this law do.
    """
    scenario = scenarios.crossing().with_manoeuvre(break_turn(7.0, start_time=0.0))
    world, _ = scenario.build(AugmentedProportionalNavigation(3.0))
    missile = world["missile"]
    assert isinstance(missile, Missile)
    world.step(0.0, 1e-3)
    world.step(1e-3, 1e-3)
    missile.update_guidance(2e-3, 0.01)

    track = missile.latest_track
    assert track is not None
    assert track.target_acceleration is not None
    assert float(np.linalg.norm(track.target_acceleration)) == pytest.approx(7.0 * G, rel=1e-6)


def test_it_transforms_a_sustained_break_turn() -> None:
    """The case APN was derived for: an acceleration that genuinely persists."""
    plain = _fly(break_turn(7.0, 8.0), ProportionalNavigation(3.0))
    augmented = _fly(break_turn(7.0, 8.0), AugmentedProportionalNavigation(3.0))
    assert plain is not None
    assert augmented is not None
    assert augmented < plain / 10.0


def test_it_recovers_the_weave_through_a_real_seeker_and_filter() -> None:
    """The whole point of the phase: a manoeuvring target hit through a seeker.

    Induced drag left plain PN missing this by 6.94 m against a 5 m lethal
    radius. APN needs an estimate of target acceleration to work, the EKF
    supplies one, and NEES established that estimate is honest — all three are
    load-bearing.
    """
    plain = [
        _fly(weave(6.0, 4.0), ProportionalNavigation(3.0), seeker=True, seed=s) for s in range(3)
    ]
    augmented = [
        _fly(weave(6.0, 4.0), AugmentedProportionalNavigation(3.0), seeker=True, seed=s)
        for s in range(3)
    ]
    assert all(m is not None for m in plain + augmented)
    plain_median = float(np.median([m for m in plain if m is not None]))
    augmented_median = float(np.median([m for m in augmented if m is not None]))

    assert augmented_median < 5.0, "APN should bring the weave inside the lethal radius"
    assert plain_median > 5.0, "plain PN should not — if it does, the comparison is stale"


def test_it_costs_nothing_against_a_target_that_does_not_manoeuvre() -> None:
    """The lead term is zero when there is nothing to lead, so nothing changes."""
    plain = _fly(straight_and_level(), ProportionalNavigation(3.0))
    augmented = _fly(straight_and_level(), AugmentedProportionalNavigation(3.0))
    assert plain is not None
    assert augmented is not None
    assert augmented == pytest.approx(plain, abs=1e-6)


# --------------------------------------------------------------------------
# Where it does not
# --------------------------------------------------------------------------
def test_a_rotating_acceleration_defeats_it_even_with_perfect_information() -> None:
    """The failure, on a perfect track, so the filter is not the explanation.

    What that failure *is* took a second look, and the answer is not the one
    this test was originally written to assert. See
    :func:`test_the_barrel_roll_is_the_airframe_not_the_assumption` below, which
    is the one that identifies the cause; this one only establishes that the
    cause is somewhere other than the estimate.
    """
    plain = _fly(barrel_roll(5.0, 4.0), ProportionalNavigation(3.0))
    augmented = _fly(barrel_roll(5.0, 4.0), AugmentedProportionalNavigation(3.0))
    assert plain is not None
    assert augmented is not None
    assert augmented > 5.0 * plain, (
        f"expected APN to lose badly here; PN {plain:.1f} m, APN {augmented:.1f} m"
    )


def _terminal_acceleration_error(manoeuvre: object) -> tuple[float, float]:
    """Median |estimated - true| target acceleration over the last 2 s, in g.

    Returns it alongside the median true acceleration, because the comparison
    that matters is between the error and the signal, not the error alone.

    Flown by hand rather than through :func:`run` because the quantity wanted
    is inside the track the guidance law was handed, and the only place that
    exists is the moment of the guidance update.
    """
    scenario = scenarios.crossing().with_manoeuvre(manoeuvre)  # type: ignore[arg-type]
    world, _ = scenario.build(
        AugmentedProportionalNavigation(3.0),
        seeker=SeekerConfig(),
        estimator=ExtendedKalman(),
        seed=0,
    )
    missile, target = world["missile"], world["target"]
    assert isinstance(missile, Missile)

    dt, guidance_interval = 1e-3, 0.01
    time, next_guidance = 0.0, 0.0
    samples: list[tuple[float, float, float]] = []
    while time < 20.0 and float(np.linalg.norm(target.state.pos - missile.state.pos)) > 5.0:
        if time >= next_guidance - 1e-9:
            missile.update_guidance(time, guidance_interval)
            next_guidance += guidance_interval
            track = missile.latest_track
            truth = target.commanded_manoeuvre(time)
            if track is not None and track.valid and track.target_acceleration is not None:
                assert truth is not None
                samples.append(
                    (
                        time,
                        float(np.linalg.norm(track.target_acceleration - truth)) / G,
                        float(np.linalg.norm(truth)) / G,
                    )
                )
        world.step(time, dt)
        time += dt

    last = samples[-1][0] - 2.0
    terminal = [(error, signal) for when, error, signal in samples if when > last]
    return (
        float(np.median([error for error, _ in terminal])),
        float(np.median([signal for _, signal in terminal])),
    )


def test_the_jink_failure_is_the_estimate_not_the_lag() -> None:
    """Why APN loses the jink through a seeker but wins it on truth.

    The obvious explanation — the filter is slow to notice each new break — is
    wrong, and was checked before this test was written: the EKF picks up a
    direction change in about 20 ms, a fiftieth of a 1.5 s segment. The real
    mechanism is that its acceleration error is *larger than the acceleration
    it is estimating*, and APN multiplies that error by N/2 on its way into the
    command. On a weave the same filter, the same seeker and the same law give
    an error a fraction of the signal, which is why the same term is worth a
    factor of six there.

    This is the distinction that decides what to do next: a manoeuvre-detecting
    estimator would help the jink, and would do nothing whatever for the barrel
    roll, which fails on a perfect track for a reason of its own.
    """
    jink_error, jink_signal = _terminal_acceleration_error(jink(7.0, 1.5))
    weave_error, weave_signal = _terminal_acceleration_error(weave(6.0, 4.0))

    assert jink_error > jink_signal, (
        f"the jink's estimate should be worse than the signal: {jink_error:.1f} g "
        f"error against {jink_signal:.1f} g of acceleration"
    )
    assert weave_error < 0.6 * weave_signal, (
        f"the weave's should not: {weave_error:.1f} g against {weave_signal:.1f} g"
    )


def _fly_with_lift(kind: str, law: str, max_lift_coefficient: float) -> float | None:
    """One engagement on truth, with a stated airframe. Miss distance in metres.

    Built from TOML rather than the scenario helpers because the airframe is
    what is being varied, and the config reader is the one place that knows how
    to spell it.
    """
    text = f"""
duration = 30.0
[missile]
position = [0.0, 0.0, 1000.0]
speed = 60.0
[missile.aero]
max_lift_coefficient = {max_lift_coefficient}
[target]
position = [0.0, 6000.0, 1000.0]
velocity = [250.0, 0.0, 0.0]
[target.manoeuvre]
{kind}
[guidance]
law = "{law}"
[seeker]
enabled = false
"""
    spec = loads(text, "airframe probe")
    world, detector = spec.build(seed=0)
    run(world, duration=spec.scenario.duration, dt=1e-3, stop=detector)
    return None if detector.result is None else detector.result.miss_distance


BARREL_ROLL = 'kind = "barrel_roll"\namplitude_g = 5.0\nperiod = 4.0'
BANKED_WEAVE = 'kind = "weave"\namplitude_g = 6.0\nperiod = 4.0\nbank_deg = 60.0'


@pytest.mark.parametrize(
    ("name", "manoeuvre"), [("barrel roll", BARREL_ROLL), ("banked weave", BANKED_WEAVE)]
)
def test_the_barrel_roll_is_the_airframe_not_the_assumption(name: str, manoeuvre: str) -> None:
    """The correction, and the most useful thing in this file.

    APN losing to a barrel roll looks exactly like its constant-acceleration
    premise failing, and that is what this file originally claimed. It is wrong.
    Hold everything else fixed and give the airframe twice the lift coefficient,
    and APN goes from twelve times worse than PN to a hundred times better —
    against the identical manoeuvre, on the identical perfect track. The premise
    is just as violated at ``Cl_max = 5`` as at 2.5.

    What actually happens is that the lead term is a request for roughly half as
    much lateral acceleration again. Where the airframe can meet it, it is worth
    a large factor. Where it saturates, the surplus is never produced — but the
    lift that *is* produced still costs induced drag, so the missile pays for
    the whole command and receives part of it, and arrives too slow to correct.

    A banked weave is the same story: the acceleration reverses exactly as it
    does in the flat weave APN wins, but the missile is already spending lift on
    holding itself up, so the same extra demand saturates.

    This is why the test exists at all. The observation that APN misses is
    cheap; knowing that a wind-tunnel number nobody has sourced is what decides
    whether the law is excellent or catastrophic is not.
    """
    cramped_pn = _fly_with_lift(manoeuvre, "pronav", 2.5)
    cramped_apn = _fly_with_lift(manoeuvre, "apn", 2.5)
    roomy_pn = _fly_with_lift(manoeuvre, "pronav", 5.0)
    roomy_apn = _fly_with_lift(manoeuvre, "apn", 5.0)
    assert None not in (cramped_pn, cramped_apn, roomy_pn, roomy_apn)
    assert cramped_pn is not None
    assert cramped_apn is not None
    assert roomy_pn is not None
    assert roomy_apn is not None

    assert cramped_apn > 5.0 * cramped_pn, (
        f"{name}: APN should lose badly on the shipped airframe; "
        f"PN {cramped_pn:.1f} m, APN {cramped_apn:.1f} m"
    )
    assert roomy_apn < roomy_pn, (
        f"{name}: with lift to spare APN should win; PN {roomy_pn:.2f} m, APN {roomy_apn:.2f} m"
    )
    assert roomy_apn < cramped_apn / 100.0, (
        f"{name}: the rescue should be dramatic, not marginal; "
        f"{cramped_apn:.1f} m becomes {roomy_apn:.2f} m"
    )


def test_the_barrel_roll_failure_is_energy_not_aim() -> None:
    """Confirms the mechanism rather than merely the outcome.

    If APN were simply aiming wrong it would arrive at a similar speed and miss.
    It arrives markedly slower, because the lift it asks for and cannot fully
    receive is paid for in induced drag regardless.
    """
    speeds = {}
    for label, law in (
        ("pn", ProportionalNavigation(3.0)),
        ("apn", AugmentedProportionalNavigation(3.0)),
    ):
        scenario = scenarios.crossing().with_manoeuvre(barrel_roll(5.0, 4.0))
        world, detector = scenario.build(law)
        result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
        speeds[label] = float(result.recorder.speed("missile")[-1])

    assert speeds["apn"] < 0.8 * speeds["pn"]
