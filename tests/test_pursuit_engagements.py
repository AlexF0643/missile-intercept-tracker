"""Pure pursuit flown against the three standard geometries.

These are the Phase 2 exit criterion, and they assert the *failure* as firmly as
the successes. Pure pursuit hitting head-on proves the guidance loop is wired up
correctly; pure pursuit missing the crossing target by tens of metres is the
result Phase 3 has to beat, and locking it in now means the improvement can be
measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import RunResult, run
from interceptor.sim.intercept import Intercept

G = 9.80665


def _fly(
    scenario: scenarios.Scenario, law: PurePursuit | None
) -> tuple[RunResult, Intercept | None]:
    world, detector = scenario.build(law)
    result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return result, detector.result


def test_pure_pursuit_intercepts_a_head_on_target() -> None:
    _, intercept = _fly(scenarios.head_on(), PurePursuit())
    assert intercept is not None
    assert intercept.hit is True
    assert intercept.miss_distance < 5.0


def test_pure_pursuit_intercepts_a_fleeing_target() -> None:
    _, intercept = _fly(scenarios.tail_chase(), PurePursuit())
    assert intercept is not None
    assert intercept.hit is True


def test_pure_pursuit_misses_a_crossing_target() -> None:
    """The whole reason proportional navigation exists.

    Pure pursuit steers at the target's present position. Against a crossing
    target that point is always behind where the target will be, so the missile
    swings into a tail chase and arrives late.
    """
    _, intercept = _fly(scenarios.crossing(), PurePursuit())
    assert intercept is not None
    assert intercept.hit is False
    assert intercept.miss_distance > 20.0


def test_the_crossing_miss_is_caused_by_saturation() -> None:
    """The failure has a signature: an impossible demand in the endgame."""
    result, _ = _fly(scenarios.crossing(), PurePursuit())
    record = result.recorder

    peak_demand = float(record.commanded("missile").max())
    peak_achieved = float(record.achieved("missile").max())

    assert peak_demand > 100.0 * G, "expected the law to ask for the impossible"
    assert peak_achieved < 30.0 * G, "the airframe cannot deliver that"
    assert peak_demand > 5.0 * peak_achieved


def test_an_unguided_missile_does_not_intercept() -> None:
    """Confirms the hits above come from the guidance, not a lucky initial aim."""
    _, intercept = _fly(scenarios.head_on(), None)
    assert intercept is None or intercept.miss_distance > 50.0


def test_the_missile_stays_within_its_airframe_limit_throughout() -> None:
    result, _ = _fly(scenarios.crossing(), PurePursuit())
    record = result.recorder

    achieved = record.achieved("missile")
    limit = record.limit("missile")
    # A small tolerance: the limit is sampled at the recording rate while the
    # command is held from the previous guidance tick.
    assert np.all(achieved <= limit + 1e-6)


@pytest.mark.parametrize("gain", [2.0, 4.0, 8.0])
def test_more_gain_does_not_rescue_the_crossing_geometry(gain: float) -> None:
    """Turning harder onto a sightline that keeps moving does not help."""
    _, intercept = _fly(scenarios.crossing(), PurePursuit(gain=gain))
    assert intercept is not None
    assert intercept.miss_distance > 5.0


def test_scenarios_are_reproducible() -> None:
    first, _ = _fly(scenarios.crossing(), PurePursuit())
    second, _ = _fly(scenarios.crossing(), PurePursuit())
    assert np.array_equal(
        first.recorder.position("missile"),
        second.recorder.position("missile"),
    )
