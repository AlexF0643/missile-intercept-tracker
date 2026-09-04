"""Filter consistency: is the EKF's stated uncertainty honest?

Every other filter test asks whether the estimate is close to the truth. These
ask whether the covariance the filter reports alongside it is a truthful account
of how close. A filter can track well and still lie about its confidence, and
the direction of the lie decides whether it is merely wasteful or actually
dangerous.

The load-bearing test in this file is the negative control at the bottom. A
consistency check that cannot fail proves nothing about the filter it passes, so
the same machinery is pointed at a deliberately sabotaged filter to show that it
catches one.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.entities.target import Target
from interceptor.sensing.consistency import (
    chi_squared_interval,
    ensemble_average,
    normalised_error_squared,
)
from interceptor.sensing.filters import AlphaBeta, ExtendedKalman, TargetEstimate
from interceptor.sensing.seeker import GeometricSeeker, SeekerConfig
from interceptor.sensing.track import FilteredTrack

RUNS = 24
"""Independent seeds per ensemble. Enough that 24 x 6 = 144 degrees of freedom
put the acceptance interval within about +/-25% of the expected value, which is
tight enough to catch a filter that is wrong by a factor of two and loose enough
not to fail on ordinary sampling noise."""

RATE = 100.0
SETTLE = 1.5
"""Seconds ignored at the start of each run. The filter initialises with a
400 m/s shrug about velocity, and NEES during that collapse says nothing about
the tuning — only that the filter has not finished admitting it was ignorant."""


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def _estimate(error: np.ndarray, covariance: np.ndarray) -> TargetEstimate:
    """An estimate that is wrong by ``error`` and claims ``covariance``."""
    return TargetEstimate(
        position=error[0:3],
        velocity=error[3:6],
        acceleration=error[6:9],
        covariance=covariance,
    )


def test_a_perfect_estimate_scores_zero() -> None:
    estimate = _estimate(np.zeros(9), np.eye(9))
    zero = np.zeros(3)
    assert normalised_error_squared(estimate, zero, zero, zero) == pytest.approx(0.0)


def test_a_one_sigma_error_in_every_state_scores_one_per_state() -> None:
    """The definition, checked directly: NEES is error measured in sigmas, squared."""
    # Estimate sits at -1 in each state, so truth at zero is one sigma away.
    estimate = _estimate(np.full(9, -1.0), np.eye(9))
    zero = np.zeros(3)
    assert normalised_error_squared(estimate, zero, zero) == pytest.approx(6.0)
    assert normalised_error_squared(estimate, zero, zero, zero) == pytest.approx(9.0)


def test_the_statistic_scales_with_the_claimed_covariance() -> None:
    """A filter claiming four times the variance reports a quarter the NEES.

    This is the whole mechanism by which overconfidence shows up: the error is
    unchanged, only the claim about it moved.
    """
    error = np.full(9, -1.0)
    zero = np.zeros(3)
    confident = normalised_error_squared(_estimate(error, np.eye(9)), zero, zero)
    humble = normalised_error_squared(_estimate(error, 4.0 * np.eye(9)), zero, zero)
    assert humble == pytest.approx(confident / 4.0)


def test_a_fixed_gain_filter_cannot_be_asked_the_question() -> None:
    """Alpha-beta has no covariance, so consistency is undefined rather than bad."""
    estimator = AlphaBeta()
    estimator.correct(
        GeometricSeeker(SeekerConfig.perfect(), np.random.default_rng(0)).measure(
            0.0,
            EntityState(pos=np.zeros(3), vel=np.array([0.0, 300.0, 0.0])),
            EntityState(pos=np.array([0.0, 4000.0, 1000.0]), vel=np.zeros(3)),
        ),
        EntityState(pos=np.zeros(3), vel=np.array([0.0, 300.0, 0.0])),
    )
    estimate = estimator.estimate()
    assert estimate is not None
    assert estimate.covariance is None
    with pytest.raises(ValueError, match="needs a covariance"):
        normalised_error_squared(estimate, np.zeros(3), np.zeros(3))


# --------------------------------------------------------------------------
# The bounds
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dof", "lower", "upper"),
    [
        # Exact quantiles from scipy.stats.chi2.ppf, kept as literals so the
        # package does not gain a scipy dependency for three constants.
        (54, 35.5863, 76.1920),
        (144, 112.6711, 179.1137),
        (225, 185.3483, 268.4378),
    ],
)
def test_the_bounds_match_exact_chi_squared_quantiles(dof: int, lower: float, upper: float) -> None:
    """Wilson-Hilferty is an approximation; this is how close it actually is."""
    computed_lower, computed_upper = chi_squared_interval(dof)
    assert computed_lower == pytest.approx(lower, rel=1e-3)
    assert computed_upper == pytest.approx(upper, rel=1e-3)


def test_the_interval_brackets_the_expected_value() -> None:
    """The mean of a chi-squared variable is its degrees of freedom."""
    for dof in (6, 54, 144, 900):
        lower, upper = chi_squared_interval(dof)
        assert lower < dof < upper


def test_more_runs_give_a_tighter_interval() -> None:
    """Why consistency needs an ensemble: the bound narrows as runs accumulate."""

    def width(runs: int) -> float:
        lower, upper = chi_squared_interval(runs * 6)
        return (upper - lower) / runs

    assert width(100) < width(24) < width(4)


@pytest.mark.parametrize("confidence", [0.5, 0.0, 1.0, 0.951])
def test_an_unsupported_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be one of"):
        chi_squared_interval(144, confidence)


def test_a_nonsense_dof_is_rejected() -> None:
    with pytest.raises(ValueError, match="dof must be at least 1"):
        chi_squared_interval(0)


# --------------------------------------------------------------------------
# The filter itself
# --------------------------------------------------------------------------
TARGET_START = np.array([-1200.0, 6000.0, 1000.0])
TARGET_VELOCITY = np.array([250.0, 0.0, 0.0])
MISSILE_START = np.array([0.0, 0.0, 1000.0])
MISSILE_VELOCITY = np.array([0.0, 600.0, 0.0])
GRAVITY = 9.80665


def _truth(t: float, weave_g: float, period: float) -> tuple[np.ndarray, np.ndarray]:
    """True target position and velocity at time ``t``.

    Straight and level when ``weave_g`` is zero, otherwise a vertical sinusoid.
    Written out analytically rather than integrated so that the comparison the
    tests make is against exact truth and not against a second simulation with
    its own error.
    """
    if weave_g == 0.0:
        return TARGET_START + TARGET_VELOCITY * t, TARGET_VELOCITY

    amplitude = weave_g * GRAVITY
    omega = 2.0 * np.pi / period
    climb = np.array([0.0, 0.0, -(amplitude / omega**2) * (np.cos(omega * t) - 1.0)])
    rate = np.array([0.0, 0.0, (amplitude / omega) * np.sin(omega * t)])
    return TARGET_START + TARGET_VELOCITY * t + climb, TARGET_VELOCITY + rate


def _track_a_target(
    seed: int,
    *,
    covariance_scale: float = 1.0,
    jerk_sigma: float = 60.0,
    weave_g: float = 0.0,
    weave_period: float = 4.0,
) -> list[float]:
    """Fly one seed and return NEES at every cycle after the settling period.

    Driven through :class:`FilteredTrack` rather than by calling the estimator
    directly, because the epoch bookkeeping that keeps measurement time and
    current time apart lives there. A hand-rolled loop would be testing a
    reimplementation of the thing under test.

    The missile flies a scripted constant velocity rather than being guided.
    That is deliberate: guidance steers towards the estimate, so a guided run
    feeds the filter's own error back into the geometry that produced it, and
    the result would be a statement about the closed loop rather than about the
    filter.

    NEES is evaluated against the truth **at the measurement's epoch**, not at
    the current time. Estimating the state at the instant it was observed is the
    filter's job; carrying that estimate forward to now is arithmetic that
    happens afterwards, and holding the filter responsible for it would be
    scoring it on someone else's work.

    Args:
        covariance_scale: Multiplies the covariance the filter reports without
            touching the gains it computes. 1.0 leaves the filter alone; smaller
            values fake overconfidence for the negative control.
    """
    target = Target("target", EntityState(pos=TARGET_START, vel=TARGET_VELOCITY))
    estimator = ExtendedKalman(jerk_sigma=jerk_sigma)
    source = FilteredTrack(
        GeometricSeeker(SeekerConfig(), np.random.default_rng(seed)), target, estimator
    )

    dt = 1.0 / RATE
    scores: list[float] = []

    for step in range(int(6.0 * RATE)):
        t = step * dt
        position, velocity = _truth(t, weave_g, weave_period)
        target.state = EntityState(pos=position, vel=velocity)
        missile = EntityState(
            pos=MISSILE_START + MISSILE_VELOCITY * t,
            vel=MISSILE_VELOCITY,
        )
        source.update(t, missile)

        measurement = source.latest
        estimate = estimator.estimate()
        if measurement is None or not measurement.valid or t < SETTLE:
            continue
        if estimate is None or estimate.covariance is None:
            continue

        observed_position, observed_velocity = _truth(measurement.time, weave_g, weave_period)
        scores.append(
            normalised_error_squared(
                TargetEstimate(
                    position=estimate.position,
                    velocity=estimate.velocity,
                    acceleration=estimate.acceleration,
                    covariance=estimate.covariance * covariance_scale,
                ),
                observed_position,
                observed_velocity,
            )
        )
    return scores


def _ensemble(**kwargs: float) -> tuple[float, tuple[float, float]]:
    """Mean NEES across independent seeds, and the interval it should sit in.

    One sample is taken per run — the last cycle — so the samples are genuinely
    independent. Averaging a whole trajectory would give correlated values and
    an interval far tighter than the data earns.
    """
    samples = np.array([_track_a_target(seed, **kwargs)[-1] for seed in range(RUNS)])
    return ensemble_average(samples)


def test_the_ekf_is_consistent() -> None:
    """The headline claim: the filter's stated uncertainty matches its real error.

    Mean NEES should land on 6, the number of states being measured. Above the
    interval means overconfident — the filter's errors are larger than it
    admits, so its gain is too small and it discounts measurements that disagree
    with it, tracking a quiet target immaculately and then arriving late on the
    manoeuvre that matters. Below means it is exaggerating its own doubt, which
    wastes performance but is safe.

    This test failed the first time it was run, at a mean of 1804 — see
    ``test_a_stale_measurement_biases_the_estimate`` in ``test_filters.py`` for
    what it caught.
    """
    mean, (lower, upper) = _ensemble()
    assert lower < mean < upper, (
        f"mean NEES {mean:.2f} is outside the consistency interval "
        f"[{lower:.2f}, {upper:.2f}] — the filter is "
        f"{'overconfident' if mean > upper else 'over-cautious'}"
    )


@pytest.mark.parametrize("jerk_sigma", [0.5, 300.0])
def test_consistency_holds_across_process_noise_settings(jerk_sigma: float) -> None:
    """Consistency is not a property of one lucky tuning.

    ``jerk_sigma`` is the filter's main knob and these two bracket the default
    by a factor of 120 either way. Both stay consistent, and the reason is worth
    understanding: inflating the process noise makes the filter trust
    measurements more, so its error grows — but its reported covariance grows
    with it, and NEES is their ratio. A filter can be badly tuned for accuracy
    while remaining perfectly honest about how badly.
    """
    mean, (lower, upper) = _ensemble(jerk_sigma=jerk_sigma)
    assert lower < mean < upper, f"jerk_sigma={jerk_sigma}: mean NEES {mean:.2f}"


def test_consistency_survives_a_target_the_model_does_not_describe() -> None:
    """A weaving target breaks the filter's assumption without breaking its honesty.

    The process model says the target holds its acceleration. A 6 g weave on a
    4 s period reverses that acceleration twice per cycle, so the model is
    continuously wrong. The white-noise-jerk process noise is what buys the
    filter the room to be wrong in — and this asserts that the room it allows
    itself is neither too small nor absurdly large.
    """
    mean, (lower, upper) = _ensemble(weave_g=6.0, weave_period=4.0)
    assert lower < mean < upper, f"weaving target: mean NEES {mean:.2f}"


# --------------------------------------------------------------------------
# The negative control
# --------------------------------------------------------------------------
def test_the_check_catches_a_filter_that_understates_its_uncertainty() -> None:
    """Does this test file actually detect the failure it claims to detect?

    Without this, every assertion above is unfalsifiable — a check that passes
    for all inputs tells you nothing about the one you fed it. So the same
    machinery is aimed at a filter sabotaged to report a covariance a hundredth
    of its real one, which is precisely the overconfidence being guarded
    against, and it has to fail.
    """
    mean, (_, upper) = _ensemble(covariance_scale=0.01)
    assert mean > upper, (
        f"a filter understating its variance 100-fold scored {mean:.2f}, still "
        f"inside the upper bound {upper:.2f} — the consistency check is not "
        f"sensitive enough to be worth running"
    )
