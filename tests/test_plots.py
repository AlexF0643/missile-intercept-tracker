"""Plotting.

Skipped wherever matplotlib is absent, since it is an optional extra. These
assert that a figure is produced with the expected structure, not what it looks
like — the visual judgement is a human's job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import RunResult, run
from interceptor.sim.intercept import Intercept
from interceptor.sim.montecarlo import Study, Trial

pytest.importorskip("matplotlib", reason="plotting is an optional extra")

from interceptor.viz.plots import (
    DARK,
    LIGHT,
    LawRun,
    plot_comparison,
    plot_engagement,
    plot_estimator_comparison,
    plot_monte_carlo,
    plot_noise_sweep,
    save_comparison,
    save_engagement,
    save_estimator_comparison,
    save_monte_carlo,
    save_noise_sweep,
)


@pytest.fixture(scope="module")
def flown() -> tuple[RunResult, Intercept | None]:
    scenario = scenarios.crossing()
    world, detector = scenario.build(PurePursuit())
    result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return result, detector.result


def test_figure_has_all_five_panels(flown: tuple[RunResult, Intercept | None]) -> None:
    result, intercept = flown
    figure = plot_engagement(result, intercept=intercept)
    assert len(figure.axes) == 5


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_both_themes_render(flown: tuple[RunResult, Intercept | None], theme: str) -> None:
    result, _ = flown
    figure = plot_engagement(result, theme=theme)
    assert figure.get_facecolor() is not None


def test_an_unknown_theme_is_rejected(flown: tuple[RunResult, Intercept | None]) -> None:
    result, _ = flown
    with pytest.raises(ValueError, match="theme must be one of"):
        plot_engagement(result, theme="solarized")


def test_saving_writes_a_png(flown: tuple[RunResult, Intercept | None], tmp_path: Path) -> None:
    result, intercept = flown
    path = save_engagement(
        result,
        tmp_path / "nested" / "engagement.png",
        intercept=intercept,
    )
    assert path.exists()
    assert path.stat().st_size > 10_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_themes_are_distinct() -> None:
    """Dark mode is a selected palette, not an inverted one."""
    assert LIGHT.surface != DARK.surface
    assert LIGHT.missile != DARK.missile
    assert LIGHT.target != DARK.target


# --------------------------------------------------------------------------
# Guidance law comparison
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_laws() -> list[LawRun]:
    """The same engagement flown by two laws, which is what the panel compares."""
    runs = []
    for label, law in (("pursuit", PurePursuit()), ("pronav", ProportionalNavigation(3.0))):
        scenario = scenarios.crossing()
        world, detector = scenario.build(law)
        result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
        runs.append(LawRun(label=label, result=result, intercept=detector.result))
    return runs


def test_the_comparison_has_a_plan_view_and_two_time_histories(two_laws: list[LawRun]) -> None:
    figure = plot_comparison(two_laws)
    assert len(figure.axes) == 3


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_comparison_renders_in_both_themes(two_laws: list[LawRun], theme: str) -> None:
    assert plot_comparison(two_laws, theme=theme).get_facecolor() is not None


def test_the_comparison_can_draw_a_single_law(two_laws: list[LawRun]) -> None:
    """A comparison of one is degenerate but should not be a crash."""
    assert len(plot_comparison(two_laws[:1]).axes) == 3


def test_the_comparison_rejects_an_unknown_theme(two_laws: list[LawRun]) -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        plot_comparison(two_laws, theme="sepia")


def test_saving_a_comparison_writes_a_png(two_laws: list[LawRun], tmp_path: Path) -> None:
    path = save_comparison(two_laws, tmp_path / "nested" / "comparison.png")
    assert path.exists()
    assert path.stat().st_size > 5_000


# --------------------------------------------------------------------------
# Noise sweep
# --------------------------------------------------------------------------
SIGMAS = [0.02, 0.06, 0.2, 0.6, 2.0]
MISSES = np.array(
    [
        [0.02, 0.03, 0.02, 0.04],
        [0.15, 0.19, 0.14, 0.22],
        [1.4, 2.1, 1.6, 1.9],
        [24.0, 31.0, 19.0, 28.0],
        [1400.0, 1800.0, 1500.0, 1600.0],
    ]
)


def test_the_noise_sweep_draws_one_panel() -> None:
    assert len(plot_noise_sweep(SIGMAS, MISSES).axes) == 1


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_noise_sweep_renders_in_both_themes(theme: str) -> None:
    assert plot_noise_sweep(SIGMAS, MISSES, theme=theme).get_facecolor() is not None


def test_the_noise_sweep_accepts_a_baseline() -> None:
    """The perfect-information reference line every degraded run is read against."""
    assert plot_noise_sweep(SIGMAS, MISSES, baseline=0.03).get_facecolor() is not None


def test_the_noise_sweep_rejects_an_unknown_theme() -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        plot_noise_sweep(SIGMAS, MISSES, theme="neon")


def test_saving_a_noise_sweep_writes_a_png(tmp_path: Path) -> None:
    path = save_noise_sweep(SIGMAS, MISSES, tmp_path / "sweep.png", baseline=0.03)
    assert path.exists()
    assert path.stat().st_size > 5_000


# --------------------------------------------------------------------------
# Estimator comparison
# --------------------------------------------------------------------------
ESTIMATORS = {
    "Straight": {"alpha-beta": np.array([0.4, 0.6, 0.5]), "EKF": np.array([0.9, 1.1, 1.0])},
    "Break turn": {"alpha-beta": np.array([9.4, 12.0, 10.1]), "EKF": np.array([2.8, 3.2, 3.0])},
}
HITS = {
    "Straight": {"alpha-beta": (3, 3), "EKF": (3, 3)},
    "Break turn": {"alpha-beta": (0, 3), "EKF": (3, 3)},
}


def test_the_estimator_comparison_draws_one_panel_per_behaviour() -> None:
    assert len(plot_estimator_comparison(ESTIMATORS).axes) == len(ESTIMATORS)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_estimator_comparison_renders_in_both_themes(theme: str) -> None:
    assert plot_estimator_comparison(ESTIMATORS, theme=theme).get_facecolor() is not None


def test_the_estimator_comparison_annotates_hit_counts() -> None:
    """Median miss alone hides the difference between a near miss and a rout."""
    figure = plot_estimator_comparison(ESTIMATORS, hits=HITS, baseline=0.03)
    labels = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert any("0/3" in label for label in labels)


def test_the_estimator_comparison_works_without_hit_counts() -> None:
    figure = plot_estimator_comparison(ESTIMATORS)
    labels = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert labels
    assert not any("/" in label for label in labels)


def test_the_estimator_comparison_survives_a_single_panel() -> None:
    """One behaviour means `subplots` returns a bare axis rather than an array."""
    single = {"Straight": ESTIMATORS["Straight"]}
    assert len(plot_estimator_comparison(single).axes) == 1


def test_every_panel_shows_its_own_data_however_far_apart_they_are() -> None:
    """The panels share one x range, and it has to cover all of them.

    This is a regression test for a figure that came out blank. The panels were
    built with ``sharex`` and then each set its own limits, so the last panel's
    won and any panel whose data was smaller fell off the left edge — silently,
    because an axis with nothing in view still draws perfectly happily. Comparing
    proportional navigation with its augmented form is what exposed it: the
    barrel-roll panel runs to 469 m while the straight-and-level panel sits at
    1 m, and three of the five panels came out empty.
    """
    spread = {
        "Tiny": {"PN": np.array([0.8, 1.2]), "APN": np.array([1.9, 2.1])},
        "Huge": {"PN": np.array([30.0, 44.0]), "APN": np.array([470.0, 600.0])},
    }
    figure = plot_estimator_comparison(spread)
    for axis in figure.axes:
        low, high = axis.get_xlim()
        assert low < 0.8, "the smallest point must be inside the view"
        assert high > 600.0, "so must the largest"


def test_the_estimator_comparison_rejects_nothing_to_compare() -> None:
    with pytest.raises(ValueError, match="nothing to compare"):
        plot_estimator_comparison({})


def test_the_estimator_comparison_rejects_an_unknown_theme() -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        plot_estimator_comparison(ESTIMATORS, theme="mono")


def test_saving_an_estimator_comparison_writes_a_png(tmp_path: Path) -> None:
    path = save_estimator_comparison(
        ESTIMATORS, tmp_path / "estimators.png", hits=HITS, baseline=0.03
    )
    assert path.exists()
    assert path.stat().st_size > 5_000


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def _study(hit_rates: list[float], seeds: int = 8) -> Study:
    """A study with a chosen probability of kill per draw, for layout tests."""
    rng = np.random.default_rng(4)
    trials = [
        Trial(
            draw=draw,
            seed=seed,
            miss_distance=float(
                rng.uniform(0.2, 4.5) if rng.random() < rate else rng.uniform(6.0, 90.0)
            ),
            hit=rng.random() < rate,
            time=13.0,
            arrival_speed=420.0,
            saturated=0.2,
            dropouts=0.0,
            reached_closest_approach=True,
        )
        for draw, rate in enumerate(hit_rates)
        for seed in range(seeds)
    ]
    return Study(trials=trials, draws=[{"missile.aero.drag_coefficient": 0.3} for _ in hit_rates])


MONTE_CARLO = {
    "PN (N=3)": _study([0.4, 0.1, 0.9, 0.5]),
    "APN (N=3)": _study([0.9, 0.2, 1.0, 0.7]),
}
SENSITIVITY = [
    ("missile.aero.drag_coefficient", -0.81),
    ("seeker.glint_sigma", -0.44),
    ("missile.aero.max_lift_coefficient", 0.22),
]


def test_the_monte_carlo_figure_has_all_three_panels() -> None:
    assert len(plot_monte_carlo(MONTE_CARLO, SENSITIVITY).axes) == 3


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_monte_carlo_figure_renders_in_both_themes(theme: str) -> None:
    assert plot_monte_carlo(MONTE_CARLO, SENSITIVITY, theme=theme).get_facecolor() is not None


def test_the_cumulative_curve_reaches_every_trial() -> None:
    """The panel's whole claim is that crossing the lethal radius is the Pk.

    That only holds if the curve is the complete sample — one step per trial,
    ending at 1.0 — so this checks the drawing rather than trusting it.
    """
    figure = plot_monte_carlo(MONTE_CARLO, SENSITIVITY)
    curves = [line for line in figure.axes[0].lines if np.size(line.get_xdata()) > 2]
    assert len(curves) == 2
    for line in curves:
        fractions = np.asarray(line.get_ydata())
        assert fractions.max() == pytest.approx(1.0)
        assert len(fractions) == len(MONTE_CARLO["PN (N=3)"].nominal_trials())


def test_the_monte_carlo_figure_survives_having_no_sensitivity() -> None:
    """A single-draw study has nothing to correlate against, and should still draw."""
    figure = plot_monte_carlo(MONTE_CARLO, [])
    assert not figure.axes[2].get_visible()


def test_the_monte_carlo_figure_rejects_nothing_to_plot() -> None:
    with pytest.raises(ValueError, match="nothing to plot"):
        plot_monte_carlo({}, SENSITIVITY)


def test_the_monte_carlo_figure_rejects_an_unknown_theme() -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        plot_monte_carlo(MONTE_CARLO, SENSITIVITY, theme="mono")


def test_saving_a_monte_carlo_figure_writes_a_png(tmp_path: Path) -> None:
    path = save_monte_carlo(MONTE_CARLO, SENSITIVITY, tmp_path / "mc.png")
    assert path.exists()
    assert path.stat().st_size > 5_000
