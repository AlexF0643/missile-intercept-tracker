"""The Monte Carlo, and the honesty of the constants it samples.

Kept fast on purpose. A study is minutes of compute and a test suite is not the
place to spend them, so everything here runs a handful of trials and checks the
*machinery*: that the sampling is reproducible, that a draw actually changes the
physics, that the statistics behave at the ends of their range, and that the
provenance table cannot quietly drift away from the schema it describes.

The findings themselves are measured by ``examples/monte_carlo.py`` and written
into the README, where a number can be re-derived rather than asserted.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

from interceptor import uncertainty
from interceptor.config import bundled_text, resolve
from interceptor.sim.montecarlo import Study, Trial, fly_one, study, wilson_interval
from interceptor.web.fields import FIELDS


@pytest.fixture(scope="module")
def crossing() -> dict[str, object]:
    return tomllib.loads(bundled_text("crossing"))


# --------------------------------------------------------------------------
# The provenance table
# --------------------------------------------------------------------------
def test_every_sampled_constant_is_a_key_a_scenario_accepts(crossing: dict[str, object]) -> None:
    """A range on a key nothing reads would vary nothing at all, silently.

    The validator rejects unknown keys, so applying every sampled constant at
    once and resolving the result is a complete check that each path is real.
    """
    drawn = uncertainty.draw(np.random.default_rng(0))
    spec = resolve(uncertainty.apply(crossing, drawn), "probe")
    assert spec.scenario.duration > 0.0


def test_the_sampled_constants_are_exactly_the_ones_admitted_to_be_guesses() -> None:
    for constant in uncertainty.CONSTANTS:
        if constant.provenance == "chosen":
            assert constant.sampled, f"{constant.path} is a guess with no stated range"
        else:
            assert not constant.sampled, f"{constant.path} is not a guess but is sampled"


def test_every_constant_says_where_it_came_from() -> None:
    for constant in uncertainty.CONSTANTS:
        assert constant.why.strip(), f"{constant.path} has no account of itself"
        assert constant.unit, f"{constant.path} has no unit"


def test_a_range_contains_its_nominal_value() -> None:
    """A range that excludes the value the project ships with would mean the
    shipped scenarios are already outside what the study calls plausible."""
    for constant in uncertainty.SAMPLED:
        assert constant.low is not None
        assert constant.high is not None
        assert constant.low < constant.value < constant.high, constant.path


def test_the_provenance_table_covers_the_settings_the_form_exposes() -> None:
    """Anything a person can drag a slider on is a number somebody chose.

    The browser form is the list of settings the project puts in front of you;
    if one of them has no entry here, it is a constant with no account of where
    it came from — which is the situation this whole file exists to end. Purely
    geometric settings are excluded: a target's starting position is the
    question being asked, not an uncertain fact about the world.
    """
    geometry = {"position", "velocity", "duration"}
    accounted = {c.path for c in uncertainty.CONSTANTS}
    missing = [
        field.path
        for field in FIELDS
        if field.path.rsplit(".", 1)[-1] not in geometry
        and field.path not in accounted
        # Manoeuvre, guidance and estimator settings describe the *scenario* or
        # the algorithm, not the missile: what the target does and which filter
        # is fitted are chosen per run, and the sweep examples explore them.
        and not field.path.startswith(("target.manoeuvre", "estimator.", "guidance."))
        and field.path not in {"seeker.enabled"}
    ]
    assert not missing, f"settings with no stated provenance: {sorted(missing)}"


def test_a_draw_is_reproducible() -> None:
    first = uncertainty.draw(np.random.default_rng(4))
    second = uncertainty.draw(np.random.default_rng(4))
    assert first == second
    assert uncertainty.draw(np.random.default_rng(5)) != first


def test_a_draw_stays_inside_the_declared_ranges() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        drawn = uncertainty.draw(rng)
        for constant in uncertainty.SAMPLED:
            assert constant.low is not None
            assert constant.high is not None
            assert constant.low <= drawn[constant.path] <= constant.high


def test_applying_a_draw_leaves_the_original_alone(crossing: dict[str, object]) -> None:
    """A thousand draws share one base scenario; mutating it would make every
    result depend on the order the trials happened to run in."""
    before = tomllib.loads(bundled_text("crossing"))
    uncertainty.apply(crossing, {"missile.aero.drag_coefficient": 0.9})
    assert crossing == before


def test_applying_a_draw_reaches_a_nested_key(crossing: dict[str, object]) -> None:
    applied = uncertainty.apply(crossing, {"missile.aero.drag_coefficient": 0.42})
    assert applied["missile"]["aero"]["drag_coefficient"] == 0.42


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_a_perfect_score_does_not_claim_certainty() -> None:
    """The reason for Wilson rather than the normal approximation.

    ``p +/- z*sqrt(p(1-p)/n)`` is exactly zero wide at 60 from 60, which would
    have this study reporting a probability of 1.000 with no interval from a
    finite sample.
    """
    low, high = wilson_interval(60, 60)
    # Analytically the upper bound at p = 1 is exactly 1; in floating point it
    # lands a bit-width below, which is not worth clamping over.
    assert high == pytest.approx(1.0)
    assert 0.9 < low < 0.98


def test_an_interval_never_leaves_the_unit_range() -> None:
    for hits, trials in [(0, 5), (1, 5), (5, 5), (0, 1), (1, 1), (3, 1000)]:
        low, high = wilson_interval(hits, trials)
        assert 0.0 <= low <= high <= 1.0


def test_more_trials_narrow_the_interval() -> None:
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_no_trials_means_no_information() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------
def test_a_study_flies_every_draw_and_seed(crossing: dict[str, object]) -> None:
    result = study(crossing, draws=2, seeds=2, workers=1, rng_seed=3)
    assert len(result.trials) == 4
    assert {(t.draw, t.seed) for t in result.trials} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(result.draws) == 2


def test_draw_zero_is_always_the_shipped_constants(crossing: dict[str, object]) -> None:
    """So that "what this project actually claims" is always in the output,
    and the epistemic layer is an addition to the conventional answer rather
    than a replacement for it."""
    result = study(crossing, draws=3, seeds=1, workers=1, rng_seed=8)
    assert result.draws[0] == uncertainty.nominal()
    assert result.draws[1] != result.draws[0]


def test_a_study_is_reproducible(crossing: dict[str, object]) -> None:
    first = study(crossing, draws=2, seeds=2, workers=1, rng_seed=11)
    second = study(crossing, draws=2, seeds=2, workers=1, rng_seed=11)
    assert [t.miss_distance for t in first.trials] == [t.miss_distance for t in second.trials]


def test_running_in_parallel_gives_the_same_answers(crossing: dict[str, object]) -> None:
    """The whole point of a process pool is that it changes only the wall clock.

    Trials are keyed by draw and seed and sorted before returning, so a result
    cannot depend on which worker finished first — which is the failure mode
    that would otherwise make a study unreproducible in a way nobody notices.
    """
    serial = study(crossing, draws=2, seeds=2, workers=1, rng_seed=5)
    parallel = study(crossing, draws=2, seeds=2, workers=2, rng_seed=5)
    assert [t.miss_distance for t in serial.trials] == [t.miss_distance for t in parallel.trials]


def test_the_constants_actually_change_the_outcome(crossing: dict[str, object]) -> None:
    """Otherwise the epistemic layer would be an expensive way to fly the same
    engagement repeatedly."""
    heavy = uncertainty.apply(crossing, {"missile.aero.drag_coefficient": 0.55})
    light = uncertainty.apply(crossing, {"missile.aero.drag_coefficient": 0.18})
    dragged = fly_one((0, 0, heavy))
    clean = fly_one((0, 0, light))
    assert dragged.arrival_speed < clean.arrival_speed
    assert dragged.miss_distance != clean.miss_distance


def test_a_trial_that_never_gets_there_is_recorded_rather_than_dropped(
    crossing: dict[str, object],
) -> None:
    """A missile too draggy to reach its target has no closest approach.

    Discarding those would quietly delete the worst outcomes from the
    distribution — the exact runs a probability of kill is supposed to count.
    """
    # Cut the run short rather than crippling the missile: the branch under
    # test is "the run ended before closest approach", and a duration limit
    # reaches it without depending on how much drag it takes to fail.
    hopeless = uncertainty.apply(crossing, {})
    hopeless["duration"] = 2.0
    trial = fly_one((0, 0, hopeless))
    assert not trial.reached_closest_approach
    assert not trial.hit
    assert trial.miss_distance > 100.0


def test_a_study_reports_a_kill_probability_per_draw(crossing: dict[str, object]) -> None:
    result = study(crossing, draws=2, seeds=3, workers=1, rng_seed=2)
    probabilities = result.kill_probabilities()
    assert len(probabilities) == 2
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_an_empty_study_answers_rather_than_dividing_by_zero() -> None:
    assert Study(trials=[]).hit_fraction == 0.0
    assert list(Study(trials=[]).by_draw()) == []


def test_the_nominal_trials_can_be_separated_out() -> None:
    trials = [
        Trial(
            draw=d,
            seed=s,
            miss_distance=1.0,
            hit=True,
            time=1.0,
            arrival_speed=1.0,
            saturated=0.0,
            dropouts=0.0,
            reached_closest_approach=True,
        )
        for d in range(3)
        for s in range(2)
    ]
    assert len(Study(trials=trials).nominal_trials()) == 2
