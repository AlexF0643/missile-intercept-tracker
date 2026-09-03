"""The seeker, and the Phase 4 exit criterion.

The most important test here is the *control*: run the whole measurement chain
with every error switched off and check the result matches the truth-data
baseline. Frame conversions are the easiest thing in this project to get subtly
wrong, and a sign error would otherwise be indistinguishable from noise.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.entities.target import Target
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.sensing.seeker import (
    GeometricSeeker,
    SeekerConfig,
    relative_position_from,
)
from interceptor.sensing.track import SeekerTrack
from interceptor.sim import scenarios
from interceptor.sim.engagement import run


def _pair(
    offset: list[float], target_velocity: list[float] | None = None
) -> tuple[EntityState, EntityState]:
    """A missile flying north at 500 m/s, and a target at ``offset`` from it."""
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]), mass=85.0)
    target = EntityState(
        pos=np.array(offset),
        vel=np.array(target_velocity if target_velocity is not None else [0.0, 0.0, 0.0]),
    )
    return missile, target


def _seeker(config: SeekerConfig, seed: int = 0) -> GeometricSeeker:
    return GeometricSeeker(config, np.random.default_rng(seed))


def _instant(**overrides: float) -> SeekerConfig:
    """A seeker config with no latency, for tests that inspect one measurement.

    The default seeker holds every reading back by a frame, so its very first
    ``measure()`` correctly reports "still settling" rather than a return.
    That is right in flight and useless for single-shot inspection.
    """
    return SeekerConfig(latency_frames=0, **overrides)


# --------------------------------------------------------------------------
# The measurement chain, errors off
# --------------------------------------------------------------------------
def test_a_perfect_seeker_measures_the_true_range() -> None:
    missile, target = _pair([0.0, 5000.0, 0.0])
    measurement = _seeker(SeekerConfig.perfect()).measure(0.0, missile, target)

    assert measurement.valid
    assert measurement.range == pytest.approx(5000.0)


def test_a_target_dead_ahead_is_on_the_boresight() -> None:
    missile, target = _pair([0.0, 5000.0, 0.0])
    measurement = _seeker(SeekerConfig.perfect()).measure(0.0, missile, target)

    assert measurement.azimuth == pytest.approx(0.0, abs=1e-12)
    assert measurement.elevation == pytest.approx(0.0, abs=1e-12)


def test_angles_have_the_expected_signs() -> None:
    """Right of the nose is positive azimuth; above it is positive elevation."""
    seeker = _seeker(SeekerConfig.perfect())

    missile, target = _pair([1000.0, 5000.0, 0.0])  # east, i.e. to the right
    assert seeker.measure(0.0, missile, target).azimuth > 0.0

    missile, target = _pair([0.0, 5000.0, 1000.0])  # above
    assert seeker.measure(0.0, missile, target).elevation > 0.0


def test_range_rate_is_negative_while_closing() -> None:
    missile, target = _pair([0.0, 5000.0, 0.0], [0.0, -200.0, 0.0])
    measurement = _seeker(SeekerConfig.perfect()).measure(0.0, missile, target)
    # Missile north at 500, target south at 200: closing at 700 m/s.
    assert measurement.range_rate == pytest.approx(-700.0)


def test_reconstruction_recovers_the_true_relative_position() -> None:
    """The round trip that the whole sensing chain depends on.

    World frame to body frame to range/azimuth/elevation and back again. If this
    drifts, every downstream result is wrong in a way that still looks smooth.
    """
    seeker = _seeker(SeekerConfig.perfect())
    rng = np.random.default_rng(7)

    for _ in range(200):
        offset = rng.uniform(-4000.0, 4000.0, size=3)
        missile, target = _pair(list(offset))
        measurement = seeker.measure(0.0, missile, target)
        if not measurement.valid:
            continue
        recovered = relative_position_from(measurement, missile)
        assert np.allclose(recovered, offset, atol=1e-6)


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------
def test_a_target_behind_the_missile_is_beyond_the_gimbal() -> None:
    missile, target = _pair([0.0, -5000.0, 0.0])
    measurement = _seeker(_instant()).measure(0.0, missile, target)

    assert not measurement.valid
    assert "gimbal" in measurement.reason


def test_a_target_just_inside_the_gimbal_limit_is_seen() -> None:
    config = _instant(gimbal_limit=np.deg2rad(40.0))
    seeker = _seeker(config)
    # 30 degrees off the nose, comfortably inside the 40 degree limit.
    offset = 5000.0 * np.array([np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0)), 0.0])
    missile, target = _pair(list(offset))

    assert seeker.measure(0.0, missile, target).valid


def test_nothing_is_detected_far_beyond_the_detection_range() -> None:
    config = _instant(detection_range=5000.0, snr_fluctuation_db=0.0)
    missile, target = _pair([0.0, 20_000.0, 0.0])
    measurement = _seeker(config).measure(0.0, missile, target)

    assert not measurement.valid
    assert "detection threshold" in measurement.reason


def test_lock_is_solid_well_inside_the_detection_range() -> None:
    config = _instant(detection_range=10_000.0)
    seeker = _seeker(config)
    missile, target = _pair([0.0, 2000.0, 0.0])

    assert all(seeker.measure(0.0, missile, target).valid for _ in range(100))


def test_signal_strength_rises_twelve_decibels_per_halving_of_range() -> None:
    """The radar equation's ``R^-4`` law, checked directly."""
    config = _instant(snr_fluctuation_db=0.0)
    seeker = _seeker(config)

    far = seeker.measure(0.0, *_pair([0.0, 4000.0, 0.0])).snr_db
    near = seeker.measure(0.0, *_pair([0.0, 2000.0, 0.0])).snr_db
    assert near - far == pytest.approx(12.04, abs=0.05)


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
def test_latency_delivers_a_measurement_one_frame_old() -> None:
    seeker = _seeker(
        SeekerConfig(
            latency_frames=1,
            angle_sigma=0.0,
            range_sigma=0.0,
            glint_sigma=0.0,
            snr_fluctuation_db=0.0,
        )
    )
    missile, _ = _pair([0.0, 0.0, 0.0])

    first = seeker.measure(0.0, missile, EntityState(np.array([0.0, 5000.0, 0.0]), np.zeros(3)))
    second = seeker.measure(0.01, missile, EntityState(np.array([0.0, 4000.0, 0.0]), np.zeros(3)))

    assert not first.valid  # nothing in the pipeline yet
    assert second.valid
    assert second.time == pytest.approx(0.0)
    assert second.range == pytest.approx(5000.0), (
        "should deliver the delayed reading, not the fresh one"
    )


def test_zero_latency_delivers_immediately() -> None:
    seeker = _seeker(SeekerConfig.perfect())
    missile, target = _pair([0.0, 5000.0, 0.0])
    measurement = seeker.measure(1.25, missile, target)

    assert measurement.valid
    assert measurement.time == pytest.approx(1.25)


# --------------------------------------------------------------------------
# Noise behaviour
# --------------------------------------------------------------------------
def test_the_same_seed_gives_the_same_measurements() -> None:
    missile, target = _pair([0.0, 5000.0, 0.0])
    first = [_seeker(_instant(), 42).measure(0.0, missile, target).range for _ in range(1)]
    second = [_seeker(_instant(), 42).measure(0.0, missile, target).range for _ in range(1)]
    assert first == second


def test_different_seeds_give_different_measurements() -> None:
    missile, target = _pair([0.0, 5000.0, 0.0])
    a = _seeker(_instant(), 1).measure(0.0, missile, target).range
    b = _seeker(_instant(), 2).measure(0.0, missile, target).range
    assert a != b


def test_cross_range_error_from_angle_noise_shrinks_as_range_falls() -> None:
    """An angular error subtends less distance the closer you get."""
    config = SeekerConfig(
        range_sigma=0.0, glint_sigma=0.0, snr_fluctuation_db=0.0, latency_frames=0
    )
    seeker = _seeker(config, seed=3)

    def spread(distance: float) -> float:
        errors = []
        for _ in range(300):
            missile, target = _pair([0.0, distance, 0.0])
            measurement = seeker.measure(0.0, missile, target)
            errors.append(measurement.azimuth * measurement.range)
        return float(np.std(errors))

    assert spread(10_000.0) > 5.0 * spread(1000.0)


def test_glint_error_grows_in_angle_as_range_falls() -> None:
    """Glint is a wander in metres, so its angular effect goes the other way.

    These two crossing over is why terminal miss distance has a floor.
    """
    config = SeekerConfig(
        range_sigma=0.0, angle_sigma=0.0, snr_fluctuation_db=0.0, latency_frames=0, glint_sigma=1.5
    )
    seeker = _seeker(config, seed=4)

    def angular_spread(distance: float) -> float:
        errors = [seeker.measure(0.0, *_pair([0.0, distance, 0.0])).azimuth for _ in range(300)]
        return float(np.std(errors))

    assert angular_spread(200.0) > 5.0 * angular_spread(5000.0)


# --------------------------------------------------------------------------
# The track source
# --------------------------------------------------------------------------
def test_a_dropout_produces_an_invalid_track() -> None:
    target = Target("target", EntityState(np.array([0.0, -5000.0, 0.0]), np.zeros(3)))
    source = SeekerTrack(_seeker(_instant()), target)
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    track = source.update(0.0, missile)
    assert not track.valid


def test_the_track_recovers_truth_when_the_seeker_is_perfect() -> None:
    target = Target("target", EntityState(np.array([0.0, 5000.0, 0.0]), np.zeros(3)))
    source = SeekerTrack(_seeker(SeekerConfig.perfect()), target)
    missile = EntityState(pos=np.zeros(3), vel=np.array([0.0, 500.0, 0.0]))

    track = source.update(0.0, missile)
    assert np.allclose(track.relative_position, [0.0, 5000.0, 0.0], atol=1e-6)


# --------------------------------------------------------------------------
# Phase 4 exit criterion
# --------------------------------------------------------------------------
def _crossing_miss(config: SeekerConfig | None, seed: int = 0) -> float | None:
    scenario = scenarios.crossing()
    world, detector = scenario.build(ProportionalNavigation(3.0), seeker=config, seed=seed)
    run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return None if detector.result is None else detector.result.miss_distance


def test_the_error_free_seeker_path_reproduces_the_truth_baseline() -> None:
    """The control. Any discrepancy here is a bug, not noise.

    Running the complete measurement chain — world to body, angles, back to
    world, differenced for velocity — with every error term zeroed must give
    what perfect information gave. It is the only way to be sure that the
    degradation measured below is caused by the noise rather than by the
    plumbing that carries it.
    """
    perfect = _crossing_miss(SeekerConfig.perfect())
    truth = _crossing_miss(None)

    assert perfect is not None
    assert truth is not None
    assert perfect < 1.0
    assert abs(perfect - truth) < 1.0


def test_a_realistic_seeker_breaks_the_intercept() -> None:
    """The Phase 4 finding. Nothing is wrong; there is simply no estimator yet.

    Relative velocity is obtained by differencing two noisy positions 10 ms
    apart, which multiplies the measurement error by a hundred. Proportional
    navigation then acts on it faithfully.
    """
    miss = _crossing_miss(SeekerConfig(), seed=0)
    assert miss is not None
    assert miss > 100.0


@pytest.mark.slow
def test_miss_distance_grows_monotonically_with_seeker_noise() -> None:
    """The Phase 4 exit criterion.

    Medians over several seeds, because one seed of a stochastic process proves
    nothing. The trend must be monotonic across every step.
    """
    base = SeekerConfig()
    medians = []
    for factor in (0.01, 0.1, 0.3, 1.0):
        misses = [_crossing_miss(base.scaled(factor), seed) for seed in range(4)]
        assert all(m is not None for m in misses)
        medians.append(float(np.median([m for m in misses if m is not None])))

    for quieter, noisier in pairwise(medians):
        assert noisier > quieter, f"not monotonic: {medians}"

    # And the span is dramatic, not marginal.
    assert medians[-1] > 100.0 * medians[0]
