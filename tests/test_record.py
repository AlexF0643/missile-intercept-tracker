"""Recording an engagement to a file, and the live viewer's one testable path.

The recording is where Phase 6's exit criterion lives, because it is the half of
the phase a machine can check. These assert that a real animated file comes out
with the right number of frames in it — not that it looks good, which is a
human's judgement, but that it is not two kilobytes of blank canvas.

Deliberately small storyboards. Rendering is the slowest thing in the suite and
a test that takes a minute is a test people stop running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from interceptor.config import loads
from interceptor.sim.engagement import RunResult, run
from interceptor.viz.scene import Storyboard, storyboard_from

pytest.importorskip("matplotlib", reason="recording is part of the optional viz extra")

from interceptor.viz.record import render_flight, save_flight

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


@pytest.fixture(scope="module")
def storyboard(flight: tuple[RunResult, object]) -> Storyboard:
    """Four frames a second, which is plenty to prove the machinery works."""
    result, intercept = flight
    return storyboard_from(result, fps=4.0, intercept=intercept, title="test flight")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Building the figure
# --------------------------------------------------------------------------
def test_rendering_returns_a_figure_and_a_way_to_draw_any_frame(
    storyboard: Storyboard,
) -> None:
    """A callable, not an animation object — see render_flight's docstring."""
    figure, draw = render_flight(storyboard)
    assert figure is not None
    draw(len(storyboard.frames) - 1)
    draw(0)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_both_themes_render(storyboard: Storyboard, theme: str) -> None:
    figure, _draw = render_flight(storyboard, theme=theme)
    assert figure.get_facecolor() is not None


def test_an_unknown_theme_is_rejected(storyboard: Storyboard) -> None:
    with pytest.raises(ValueError, match="theme must be one of"):
        render_flight(storyboard, theme="chrome")


def test_an_empty_storyboard_is_rejected(storyboard: Storyboard) -> None:
    empty = Storyboard(
        frames=[], fps=30.0, bounds=storyboard.bounds, intercept=None, title="nothing"
    )
    with pytest.raises(ValueError, match="no frames"):
        render_flight(empty)


def test_the_vertical_exaggeration_is_stated_on_the_axis(storyboard: Storyboard) -> None:
    """An air engagement is nearly flat, so the vertical has to be stretched.

    Stretching it silently would be a lie about the geometry; the factor belongs
    on the label where a reader can see it.
    """
    figure, _draw = render_flight(storyboard)
    assert "scale x" in figure.axes[0].get_zlabel()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Writing a file
# --------------------------------------------------------------------------
def test_saving_writes_an_animated_gif(storyboard: Storyboard, tmp_path: Path) -> None:
    """Every frame must reach the file, which is the failure mode worth catching."""
    from PIL import Image

    path = save_flight(storyboard, tmp_path / "nested" / "flight.gif", dpi=60)
    assert path.exists()

    with Image.open(path) as image:
        assert image.n_frames == len(storyboard.frames)  # type: ignore[attr-defined]
    assert path.stat().st_size > 10_000


def test_saving_creates_missing_directories(storyboard: Storyboard, tmp_path: Path) -> None:
    path = save_flight(storyboard, tmp_path / "a" / "b" / "c" / "flight.gif", dpi=50)
    assert path.exists()


def test_asking_for_mp4_without_ffmpeg_says_so(
    storyboard: Storyboard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing external binary should suggest the alternative, not raise from
    deep inside matplotlib.

    ffmpeg's absence is faked rather than skipped around. The first version of
    this test skipped wherever ffmpeg happened to be installed, which meant it
    never ran on the machine it was written on and a real bug went out: the
    animation was being built before the writer was checked, so the error came
    with a stray "Animation was deleted without rendering" warning attached to
    whatever unrelated test the garbage collector interrupted. A test that only
    runs on other people's machines is not a test.
    """
    from matplotlib.animation import FFMpegWriter

    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        save_flight(storyboard, tmp_path / "flight.mp4")

    # Nothing should have been created on the way to that error.
    assert not (tmp_path / "flight.mp4").exists()


# --------------------------------------------------------------------------
# The live viewer
# --------------------------------------------------------------------------
def test_the_live_viewer_explains_how_to_install_vpython(
    storyboard: Storyboard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path through the live viewer that can be tested without a display.

    VPython is an optional extra, so the common experience of this module is
    someone who has not installed it. That should be a sentence telling them
    what to type, not an ImportError traceback — and it is the only part of
    ``live.py`` a machine can check, since the rest opens a window.
    """
    from interceptor.viz.live import view

    monkeypatch.setitem(sys.modules, "vpython", None)
    with pytest.raises(RuntimeError, match="pip install"):
        view(storyboard)


def test_the_live_viewer_rejects_an_empty_storyboard(storyboard: Storyboard) -> None:
    from interceptor.viz.live import view

    empty = Storyboard(
        frames=[], fps=30.0, bounds=storyboard.bounds, intercept=None, title="nothing"
    )
    with pytest.raises(ValueError, match="no frames"):
        view(empty)


def test_importing_the_live_viewer_does_not_need_vpython() -> None:
    """The module must import on a headless machine.

    VPython starts a web server and opens a browser tab when *it* is imported,
    which is why this module's import of it is inside a function. If that ever
    moves to the top of the file, every headless run pays for a viewer nobody
    asked for — and this test fails.
    """
    import importlib

    module = importlib.import_module("interceptor.viz.live")
    assert hasattr(module, "view")
    assert "vpython" not in sys.modules or sys.modules["vpython"] is None
