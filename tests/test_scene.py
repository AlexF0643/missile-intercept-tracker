"""The scene model — the part of Phase 6 with arithmetic in it.

A live 3D window cannot be asserted about on a build server, so everything that
could be wrong by more than a pixel was pushed into this module and is checked
here: the resampling, the derived numbers on the readout, and the bounds both
renderers set their axes from.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from interceptor.config import loads
from interceptor.sim.engagement import RunResult, run
from interceptor.viz.scene import GRAVITY, storyboard_from

SHORT = """
name = "short"
duration = 8.0

[missile]
position = [0.0, 0.0, 1000.0]
speed = 60.0

[target]
position = [0.0, 3500.0, 1000.0]
velocity = [220.0, 0.0, 40.0]

[guidance]
law = "pronav"

[seeker]
enabled = false
"""


@pytest.fixture(scope="module")
def flight() -> tuple[RunResult, object]:
    spec = loads(SHORT)
    world, detector = spec.build(seed=0)
    result = run(world, duration=spec.scenario.duration, dt=1e-3, stop=detector)
    return result, detector.result


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def test_the_frame_count_follows_the_frame_rate(flight: tuple[RunResult, object]) -> None:
    result, _ = flight
    slow = storyboard_from(result, fps=10.0)
    fast = storyboard_from(result, fps=30.0)
    assert len(fast.frames) == pytest.approx(3 * len(slow.frames), rel=0.05)


def test_frames_advance_in_time(flight: tuple[RunResult, object]) -> None:
    times = [frame.time for frame in storyboard_from(flight[0], fps=20.0).frames]
    assert all(later >= earlier for earlier, later in pairwise(times))


def test_frames_are_evenly_spaced_in_simulated_time(flight: tuple[RunResult, object]) -> None:
    """A wobble here is an animation that speeds up and slows down for no reason."""
    times = np.array([frame.time for frame in storyboard_from(flight[0], fps=20.0).frames])
    gaps = np.diff(times)
    # Frames snap to recorded samples, which arrive every 5 ms, so a gap can be
    # one sample either side of the nominal 50 ms. The final gap is excluded:
    # the run does not end on an exact frame boundary, so the last interval is a
    # partial one by construction rather than by mistake.
    assert float(np.abs(gaps[:-1] - 0.05).max()) <= 0.006


def test_halving_the_playback_speed_doubles_the_frames(
    flight: tuple[RunResult, object],
) -> None:
    """Slow motion means more frames over the same interval, not longer-held ones."""
    result, _ = flight
    normal = storyboard_from(result, fps=20.0, speed=1.0)
    slowed = storyboard_from(result, fps=20.0, speed=0.5)
    assert len(slowed.frames) == pytest.approx(2 * len(normal.frames), rel=0.05)
    assert slowed.duration == pytest.approx(normal.duration, rel=0.02)


def test_slow_motion_makes_the_endgame_denser(flight: tuple[RunResult, object]) -> None:
    """The whole reason the feature exists: the last seconds get more frames."""
    result, _ = flight
    boundary = 5.0
    storyboard = storyboard_from(result, fps=20.0, slow_motion_from=boundary, slow_factor=0.25)

    times = np.array([frame.time for frame in storyboard.frames])
    before = float(np.median(np.diff(times[times < boundary])))
    after = float(np.median(np.diff(times[times > boundary])))
    # A quarter, give or take the 5 ms recording grid the frames snap to — at
    # 12.5 ms nominal, one sample of rounding is 40%.
    assert after < before * 0.5
    assert after == pytest.approx(before * 0.25, rel=0.45)


def test_trails_grow_and_end_at_the_current_position(flight: tuple[RunResult, object]) -> None:
    storyboard = storyboard_from(flight[0], fps=15.0)
    first, last = storyboard.frames[0], storyboard.frames[-1]
    assert len(last.missile_trail) > len(first.missile_trail)
    assert last.missile_trail[-1] == pytest.approx(last.missile_position)
    assert last.target_trail[-1] == pytest.approx(last.target_position)


def test_at_finds_the_nearest_frame(flight: tuple[RunResult, object]) -> None:
    storyboard = storyboard_from(flight[0], fps=20.0)
    assert storyboard.at(3.0).time == pytest.approx(3.0, abs=0.05)


# --------------------------------------------------------------------------
# Derived numbers
# --------------------------------------------------------------------------
def test_the_range_matches_the_two_positions(flight: tuple[RunResult, object]) -> None:
    for frame in storyboard_from(flight[0], fps=5.0).frames:
        separation = float(np.linalg.norm(frame.target_position - frame.missile_position))
        assert frame.range == pytest.approx(separation)


def test_closing_speed_is_positive_while_the_range_falls(
    flight: tuple[RunResult, object],
) -> None:
    frames = storyboard_from(flight[0], fps=10.0).frames
    for earlier, later in pairwise(frames):
        if later.range < earlier.range - 1.0:
            assert earlier.closing_speed > 0.0


def test_the_line_of_sight_rate_is_reported(flight: tuple[RunResult, object]) -> None:
    """Non-zero and finite. It is the number the whole project exists to drive."""
    rates = [frame.los_rate for frame in storyboard_from(flight[0], fps=10.0).frames]
    assert all(np.isfinite(rate) for rate in rates)
    assert max(rates) > 0.0


def test_accelerations_are_reported_in_g(flight: tuple[RunResult, object]) -> None:
    """A readout in m/s^2 would mean nothing to anybody watching.

    Checked against the recorded acceleration rather than against a constant:
    the claim is that the conversion happened, not that 9.80665 is 9.80665.
    """
    result, _ = flight
    # 200 fps against a 200 Hz recording means every sample becomes a frame, so
    # the two peaks are the same instant rather than nearly the same instant.
    storyboard = storyboard_from(result, fps=200.0)
    peak_in_g = max(frame.achieved_g for frame in storyboard.frames)
    peak_in_si = float(np.max(result.recorder.achieved("missile")))
    assert peak_in_g == pytest.approx(peak_in_si / GRAVITY, rel=1e-6)
    assert 0.5 < peak_in_g < 40.0


def test_the_bounds_contain_every_point(flight: tuple[RunResult, object]) -> None:
    """Both renderers set their axes from this, so anything outside is off-screen."""
    storyboard = storyboard_from(flight[0], fps=10.0)
    lower, upper = storyboard.bounds
    for frame in storyboard.frames:
        assert np.all(frame.missile_position >= lower - 1e-6)
        assert np.all(frame.missile_position <= upper + 1e-6)
        assert np.all(frame.target_position >= lower - 1e-6)
        assert np.all(frame.target_position <= upper + 1e-6)


# --------------------------------------------------------------------------
# Rejection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fps", [0.0, -5.0])
def test_a_nonsense_frame_rate_is_rejected(flight: tuple[RunResult, object], fps: float) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        storyboard_from(flight[0], fps=fps)


def test_a_nonsense_speed_is_rejected(flight: tuple[RunResult, object]) -> None:
    with pytest.raises(ValueError, match="speed must be positive"):
        storyboard_from(flight[0], speed=0.0)


@pytest.mark.parametrize("factor", [0.0, -1.0, 1.5])
def test_a_nonsense_slow_factor_is_rejected(
    flight: tuple[RunResult, object], factor: float
) -> None:
    with pytest.raises(ValueError, match="slow_factor must be in"):
        storyboard_from(flight[0], slow_motion_from=2.0, slow_factor=factor)
