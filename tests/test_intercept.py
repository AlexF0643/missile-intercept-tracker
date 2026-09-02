"""Closest approach and miss distance.

The measurement this project's every later comparison rests on, so it is tested
against geometry with a closed-form answer rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.core.state import EntityState
from interceptor.core.world import World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import Target
from interceptor.sim.engagement import run
from interceptor.sim.intercept import ClosestApproach

DT = 1e-3


def _drifting_world(offset: float, start_range: float = 1000.0) -> tuple[World, ClosestApproach]:
    """Two bodies on straight lines that pass with a known separation.

    No gravity, no drag, no guidance — pure kinematics, so the closest approach
    is exactly ``offset`` metres at exactly ``start_range / 500`` seconds.

    Missile: from the origin at 300 m/s east.
    Target:  from (start_range, offset) at 200 m/s west.
    Closing at 500 m/s.
    """
    world = World(WorldConfig(enable_gravity=False, enable_drag=False))
    world.add(Missile("missile", EntityState(pos=np.zeros(3), vel=np.array([300.0, 0.0, 0.0]))))
    world.add(
        Target(
            "target",
            EntityState(pos=np.array([start_range, offset, 0.0]), vel=np.array([-200.0, 0.0, 0.0])),
        )
    )
    return world, ClosestApproach("missile", "target", lethal_radius=5.0)


def test_miss_distance_matches_the_analytic_value() -> None:
    world, detector = _drifting_world(offset=50.0)
    run(world, duration=5.0, dt=DT, stop=detector)

    assert detector.result is not None
    assert detector.result.miss_distance == pytest.approx(50.0, abs=1e-6)
    assert detector.result.time == pytest.approx(2.0, abs=1e-3)
    assert detector.result.closing_speed == pytest.approx(500.0, abs=1e-6)
    assert detector.result.hit is False


def test_interpolation_beats_reading_the_smallest_sample() -> None:
    """The reason this class exists rather than ``separation.min()``.

    A perfect collision course, with the start range offset by a quarter metre
    so the crossing falls midway *between* two samples rather than on one. The
    true miss distance is zero; the nearest sample is half a millisecond away,
    and at 500 m/s of closing speed that is a quarter of a metre.

    A quarter of a metre is the same order as the difference between a good
    guidance law and a mediocre one, so without this interpolation every
    comparison from Phase 3 onward would be measuring the sampling grid.
    """
    world, detector = _drifting_world(offset=0.0, start_range=1000.25)
    result = run(world, duration=5.0, dt=DT, record_hz=1000.0, stop=detector)

    sampled_minimum = float(result.recorder.separation("missile", "target").min())

    assert detector.result is not None
    assert detector.result.miss_distance < 1e-9
    assert sampled_minimum == pytest.approx(0.25, abs=0.01)


def test_a_close_pass_counts_as_a_hit() -> None:
    world, detector = _drifting_world(offset=3.0)
    run(world, duration=5.0, dt=DT, stop=detector)

    assert detector.result is not None
    assert detector.result.hit is True
    assert "HIT" in str(detector.result)


def test_a_wide_pass_counts_as_a_miss() -> None:
    world, detector = _drifting_world(offset=25.0)
    run(world, duration=5.0, dt=DT, stop=detector)

    assert detector.result is not None
    assert detector.result.hit is False
    assert "miss" in str(detector.result)


def test_the_run_stops_at_closest_approach() -> None:
    world, detector = _drifting_world(offset=50.0)
    result = run(world, duration=30.0, dt=DT, stop=detector)

    assert result.reason == "closest approach"
    assert result.end_time == pytest.approx(2.0, abs=1e-2)


def test_no_result_when_the_bodies_never_close() -> None:
    """Two bodies separating from the start never reach a closest approach."""
    world = World(WorldConfig(enable_gravity=False, enable_drag=False))
    world.add(Missile("missile", EntityState(pos=np.zeros(3), vel=np.array([-300.0, 0.0, 0.0]))))
    world.add(
        Target(
            "target",
            EntityState(pos=np.array([1000.0, 0.0, 0.0]), vel=np.array([200.0, 0.0, 0.0])),
        )
    )
    detector = ClosestApproach("missile", "target")

    result = run(world, duration=5.0, dt=DT, stop=detector)
    assert detector.result is None
    assert result.reason == "duration elapsed"
