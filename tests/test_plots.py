"""Plotting.

Skipped wherever matplotlib is absent, since it is an optional extra. These
assert that a figure is produced with the expected structure, not what it looks
like — the visual judgement is a human's job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import RunResult, run
from interceptor.sim.intercept import Intercept

pytest.importorskip("matplotlib", reason="plotting is an optional extra")

from interceptor.viz.plots import DARK, LIGHT, plot_engagement, save_engagement


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
