"""Estimators, and the Phase 5 exit criterion.

The criterion is stated as a comparison rather than an absolute: the filtered
intercept must land inside the lethal radius where the unfiltered one missed by
more than a kilometre. An absolute threshold would be arbitrary — angle noise
and glint put a floor under terminal miss distance that no filter removes.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.entities.target import Target, break_turn, weave
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.sensing.filters import AlphaBeta, Estimator, ExtendedKalman
from interceptor.sensing.seeker import GeometricSeeker, Measurement, Seeker, SeekerConfig
from interceptor.sensing.track import FilteredTrack
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

LETHAL_RADIUS = 5.0


def _fly(
    estimator: Estimator | None,
    *,
    manoeuvre: object = None,
    seed: int = 0,
    seeker: SeekerConfig | None = None,
) -> float | None:
    scenario = scenarios.crossing()
    if manoeuvre is not None:
        scenario = scenario.with_manoeuvre(manoeuvre)  # type: ignore[arg-type]
    world, detector = scenario.build(
        ProportionalNavigation(3.0),
        seeker=seeker if seeker is not None else SeekerConfig(),
        estimator=estimator,
        seed=seed,
    )
    run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return None if detector.result is None else detector.result.miss_distance


# --------------------------------------------------------------------------
# Alpha-beta
# --------------------------------------------------------------------------
def test_alpha_beta_derives_a_critically_damped_beta() -> None:
    """``beta = alpha^2 / (2 - alpha)`` is the fastest non-ringing pairing."""
    assert AlphaBeta(alpha=0.5).beta == pytest.approx(0.25 / 1.5)


def test_alpha_beta_accepts_an_explicit_beta() -> None:
    assert AlphaBeta(alpha=0.5, beta=0.1).beta == pytest.approx(0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, 1.5])
def test_alpha_beta_rejects_gains_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        AlphaBeta(alpha=alpha)


def test_an_estimator_reports_nothing_before_its_first_measurement() -> None:
    assert AlphaBeta().estimate() is None
    assert ExtendedKalman().estimate() is None


def test_alpha_beta_converges_on_a_constant_velocity_target() -> None:
    """Fed clean measurements of a straight-flying target, it should find the velocity."""
    truth_velocity = np.array([250.0, -100.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), truth_velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    source = FilteredTrack(
        GeometricSeeker(SeekerConfig.perfect(), np.random.default_rng(0)),
        target,
        AlphaBeta(alpha=0.3),
    )

    dt = 0.01
    for step in range(400):
        t = step * dt
        target.state = EntityState(target.state.pos + truth_velocity * dt, truth_velocity)
        source.update(t, missile)

    estimate = source.estimator.estimate()
    assert estimate is not None
    assert np.allclose(estimate.velocity, truth_velocity, atol=5.0)


# --------------------------------------------------------------------------
# Extended Kalman filter
# --------------------------------------------------------------------------
def test_the_ekf_converges_on_a_constant_velocity_target() -> None:
    truth_velocity = np.array([250.0, -100.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), truth_velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    source = FilteredTrack(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(1)),
        target,
        ExtendedKalman(),
    )

    dt = 0.01
    for step in range(600):
        t = step * dt
        target.state = EntityState(target.state.pos + truth_velocity * dt, truth_velocity)
        source.update(t, missile)

    estimate = source.estimator.estimate()
    assert estimate is not None
    assert np.allclose(estimate.velocity, truth_velocity, atol=25.0)


def test_the_ekf_covariance_falls_then_settles() -> None:
    """Uncertainty should collapse as evidence arrives, then stop.

    Two claims, and the second matters as much as the first. Velocity
    uncertainty starts at whatever the initialisation guessed (400 m/s, a
    deliberate shrug) and falls by orders of magnitude within a second or so of
    measurements. It then stops falling, because the process noise puts back on
    every prediction step roughly what the correction takes off — the filter has
    conceded that the target may be manoeuvring and refuses to become more
    confident than that admission allows.

    A covariance that kept shrinking would be the classic filter-divergence
    failure: growing certainty in an estimate that is free to drift, until the
    filter rejects the very measurements that would correct it.
    """
    truth_velocity = np.array([250.0, 0.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), truth_velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    filter_ = ExtendedKalman()
    source = FilteredTrack(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(2)), target, filter_
    )

    traces = []
    dt = 0.01
    for step in range(300):
        target.state = EntityState(target.state.pos + truth_velocity * dt, truth_velocity)
        source.update(step * dt, missile)
        estimate = filter_.estimate()
        if estimate is not None and estimate.covariance is not None:
            traces.append(float(np.trace(estimate.covariance[3:6, 3:6])))

    # Orders of magnitude, not percentages: 400 m/s of initial doubt against a
    # floor in the tens of m/s.
    assert traces[-1] < traces[0] / 100.0

    # And then flat. Compare the last two thirds of the run against each other
    # rather than testing for exact equality, which no stochastic filter owes.
    settled = traces[len(traces) // 3 :]
    assert max(settled) / min(settled) < 1.5


def test_the_ekf_estimates_target_acceleration() -> None:
    """The capability augmented proportional navigation will need in Phase 7.

    The alpha-beta filter has no acceleration state and reports zero; the EKF
    has one and should find a real manoeuvre, even if it lags it.
    """
    acceleration = np.array([0.0, 0.0, 0.0])
    acceleration[0] = 5.0 * 9.80665
    velocity = np.array([200.0, 0.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    filter_ = ExtendedKalman(jerk_sigma=80.0)
    source = FilteredTrack(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(3)), target, filter_
    )

    dt = 0.01
    state = target.state
    for step in range(600):
        state = EntityState(
            state.pos + state.vel * dt + 0.5 * acceleration * dt * dt,
            state.vel + acceleration * dt,
        )
        target.state = state
        source.update(step * dt, missile)

    estimate = filter_.estimate()
    assert estimate is not None
    # Within a factor of two of the truth is enough: the point is that it sees a
    # manoeuvre at all, in the right direction, rather than reporting zero.
    assert estimate.acceleration[0] > 0.4 * acceleration[0]
    assert estimate.acceleration[0] < 2.0 * acceleration[0]


def test_alpha_beta_reports_no_acceleration() -> None:
    """It has no state for one, and says so rather than guessing."""
    target = Target("target", EntityState(np.array([0.0, 5000.0, 0.0]), np.zeros(3)))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))
    source = FilteredTrack(
        GeometricSeeker(SeekerConfig.perfect(), np.random.default_rng(0)), target, AlphaBeta()
    )
    source.update(0.0, missile)

    estimate = source.estimator.estimate()
    assert estimate is not None
    assert np.allclose(estimate.acceleration, 0.0)


# --------------------------------------------------------------------------
# Coasting through a dropout
# --------------------------------------------------------------------------
class _Blackout(Seeker):
    """Wraps a seeker and blinds it for a window, to test coasting."""

    def __init__(self, inner: Seeker, start: float, stop: float) -> None:
        self.inner = inner
        self.start = start
        self.stop = stop

    def measure(self, t: float, missile: EntityState, target: EntityState) -> Measurement:
        if self.start <= t < self.stop:
            return Measurement.dropout(t, "blackout")
        return self.inner.measure(t, missile, target)


def test_the_track_stays_valid_through_a_half_second_blackout() -> None:
    """The Phase 5 dropout criterion.

    The naive Phase 4 track throws its history away on a dropout and reports an
    invalid track. A filter keeps predicting, so the guidance law continues to
    be served — degraded, but served.
    """
    velocity = np.array([250.0, 0.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    seeker = _Blackout(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(5)), start=2.0, stop=2.5
    )
    source = FilteredTrack(seeker, target, ExtendedKalman())

    dt = 0.01
    valid_during_blackout = 0
    for step in range(400):
        t = step * dt
        target.state = EntityState(target.state.pos + velocity * dt, velocity)
        track = source.update(t, missile)
        if 2.0 <= t < 2.5:
            valid_during_blackout += int(track.valid)

    assert valid_during_blackout == 50, "the filter should coast, not go blind"
    assert source.dropouts >= 50


def test_coasting_keeps_the_estimate_close_to_truth() -> None:
    """Half a second of prediction on a constant-velocity model should barely drift."""
    velocity = np.array([250.0, 0.0, 0.0])
    target = Target("target", EntityState(np.array([0.0, 5000.0, 1000.0]), velocity))
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    seeker = _Blackout(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(6)), start=3.0, stop=3.5
    )
    filter_ = ExtendedKalman()
    source = FilteredTrack(seeker, target, filter_)

    dt = 0.01
    for step in range(360):  # up to t = 3.6 s, just past the blackout
        target.state = EntityState(target.state.pos + velocity * dt, velocity)
        source.update(step * dt, missile)

    estimate = filter_.estimate()
    assert estimate is not None
    error = float(np.linalg.norm(estimate.position - target.state.pos))
    assert error < 50.0, f"drifted {error:.1f} m across a 0.5 s blackout"


# --------------------------------------------------------------------------
# Phase 5 exit criterion
# --------------------------------------------------------------------------
def test_a_filter_turns_a_kilometre_miss_into_a_hit() -> None:
    """The headline result. Same seeker, same guidance law, an estimator between."""
    unfiltered = _fly(None)
    filtered = _fly(ExtendedKalman())

    assert unfiltered is not None
    assert filtered is not None
    assert unfiltered > 1000.0, "expected the Phase 4 failure to still be there"
    assert filtered < LETHAL_RADIUS, f"filtered miss was {filtered:.2f} m"
    assert filtered < unfiltered / 100.0


@pytest.mark.parametrize(
    "estimator_factory",
    [lambda: AlphaBeta(alpha=0.2), lambda: ExtendedKalman(jerk_sigma=60.0)],
    ids=["alpha-beta", "ekf"],
)
def test_both_filters_intercept_a_straight_target(estimator_factory: object) -> None:
    miss = _fly(estimator_factory())  # type: ignore[operator]
    assert miss is not None
    assert miss < LETHAL_RADIUS


@pytest.mark.slow
def test_the_ekf_holds_up_against_a_manoeuvring_target_where_a_tuned_gain_fails() -> None:
    """The argument for the EKF, stated as the comparison that actually makes it.

    A heavily smoothed alpha-beta filter is the *best* estimator against a
    straight target and the *worst* against a hard turn — its lag is fatal. The
    EKF is never quite the best and never bad, which matters because a real
    engagement does not tell you in advance what the target will do.
    """
    turn = break_turn(7.0, 8.0)
    smoothed = [_fly(AlphaBeta(alpha=0.05), manoeuvre=turn, seed=s) for s in range(4)]
    kalman = [_fly(ExtendedKalman(jerk_sigma=60.0), manoeuvre=turn, seed=s) for s in range(4)]

    assert all(m is not None for m in smoothed)
    assert all(m is not None for m in kalman)

    smoothed_median = float(np.median([m for m in smoothed if m is not None]))
    kalman_median = float(np.median([m for m in kalman if m is not None]))

    assert kalman_median < smoothed_median
    assert kalman_median < LETHAL_RADIUS
    assert smoothed_median > LETHAL_RADIUS


@pytest.mark.slow
def test_a_weaving_target_costs_every_estimator_accuracy() -> None:
    """No filter setting escapes the trade; a manoeuvre always costs something."""
    straight = float(np.median([_fly(ExtendedKalman(), seed=s) or 0.0 for s in range(4)]))
    weaving = float(
        np.median(
            [_fly(ExtendedKalman(), manoeuvre=weave(6.0, 4.0), seed=s) or 0.0 for s in range(4)]
        )
    )
    assert weaving > straight
