"""Engagement diagnostics.

Five panels, each answering a question you will actually ask while debugging a
guidance law:

1. **Plan view** — the shape of the intercept from above. This is where pure
   pursuit's tail chase and proportional navigation's straight lead course look
   obviously different, which no single number conveys.
2. **Side view** — altitude against ground distance, so a missile quietly
   flying into the ground is visible.
3. **Separation** — range against time on a log scale, because the interesting
   part spans four decades from six kilometres to a few metres, and a linear
   axis throws away everything that matters at the end.
4. **Speed** — how much energy the missile has left when it arrives. A guidance
   law that intercepts while slow has not really succeeded.
5. **Lateral acceleration** — commanded, achieved, and the airframe's limit on
   one axis. The gap between the first two is saturation, and it is the single
   most diagnostic trace in the set.

Matplotlib is an optional dependency (``pip install -e ".[viz]"``), and nothing
else in the package imports this module, so a headless machine or a CI runner
can do everything except draw.

Colour: two categorical hues, checked with the OKLab/CVD validator rather than
by eye — orange for the missile, blue for the target, at ΔE 24.7 under protanopia
and 33.6 under normal vision. Identity is never carried by colour alone; every
series is also directly labelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from interceptor.core.state import Vector
from interceptor.sim.engagement import RunResult
from interceptor.sim.intercept import Intercept
from interceptor.sim.recorder import Recorder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "LawRun",
    "Theme",
    "plot_comparison",
    "plot_engagement",
    "plot_noise_sweep",
    "save_comparison",
    "save_engagement",
    "save_noise_sweep",
]

_G: Final = 9.80665


@dataclass(frozen=True)
class Theme:
    """Colours for one rendering mode. Both are validated against their surface.

    ``laws`` is the fixed order in which guidance laws are coloured when several
    are compared on one figure. Slots are assigned by position and never cycled.

    There are deliberately only two. Adding a third means finding a hue that
    separates from *both* the target's blue and the first law's orange, in both
    modes, across all pairs rather than adjacent ones — and the obvious
    candidates fail: violet collides with blue under protanopia (ΔE 1.9 in dark
    mode), magenta and yellow both collide with orange for normal vision. Aqua
    survives, at the cost of a light-mode contrast warning that the direct
    labels on every series discharge. A fourth slot needs the validator run
    again, not a hue picked by eye.
    """

    surface: str
    text: str
    muted: str
    grid: str
    missile: str
    target: str
    limit: str
    laws: tuple[str, ...]


LIGHT = Theme(
    surface="#fcfcfb",
    text="#0b0b0b",
    muted="#52514e",
    grid="#e4e4e0",
    missile="#eb6834",
    target="#2a78d6",
    limit="#8a8a85",
    laws=("#eb6834", "#1baf7a"),
)

DARK = Theme(
    surface="#1a1a19",
    text="#ffffff",
    muted="#c3c2b7",
    grid="#333330",
    missile="#d95926",
    target="#3987e5",
    limit="#7e7e78",
    laws=("#d95926", "#199e70"),
)

THEMES: Final[dict[str, Theme]] = {"light": LIGHT, "dark": DARK}


def _style(ax: Axes, theme: Theme, xlabel: str, ylabel: str, title: str) -> None:
    """Recessive axes and grid, so the data is the only assertive thing."""
    ax.set_facecolor(theme.surface)
    ax.set_title(title, color=theme.text, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel, color=theme.muted, fontsize=9)
    ax.set_ylabel(ylabel, color=theme.muted, fontsize=9)
    ax.tick_params(colors=theme.muted, labelsize=8.5)
    ax.grid(visible=True, color=theme.grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in {"left", "bottom"})
        spine.set_color(theme.grid)


def plot_engagement(
    result: RunResult,
    *,
    missile: str = "missile",
    target: str = "target",
    intercept: Intercept | None = None,
    title: str | None = None,
    theme: str = "light",
) -> Figure:
    """Build the five-panel diagnostic figure for one run.

    Args:
        result: What :func:`~interceptor.sim.engagement.run` returned.
        missile: Name of the pursuing entity in the recorder.
        target: Name of the pursued entity.
        intercept: Closest-approach result, if one was measured. Marks the point
            of closest approach and reports the miss distance.
        title: Figure heading. Defaults to a plain description.
        theme: ``"light"`` or ``"dark"``.

    Returns:
        The matplotlib figure, so the caller can save it or show it.
    """
    import matplotlib.pyplot as plt

    if theme not in THEMES:
        msg = f"theme must be one of {sorted(THEMES)}, got {theme!r}"
        raise ValueError(msg)
    palette = THEMES[theme]

    record = result.recorder
    time = record.time
    missile_pos = record.position(missile)
    target_pos = record.position(target)

    figure = plt.figure(figsize=(12.0, 11.0), dpi=110, facecolor=palette.surface)
    grid = figure.add_gridspec(3, 2, hspace=0.42, wspace=0.26, top=0.90, bottom=0.06)

    _plan_view(figure.add_subplot(grid[0, 0]), palette, missile_pos, target_pos, intercept)
    _side_view(figure.add_subplot(grid[0, 1]), palette, missile_pos, target_pos)
    _separation(figure.add_subplot(grid[1, 0]), palette, time, record, missile, target, intercept)
    _speed(figure.add_subplot(grid[1, 1]), palette, time, record, missile, target)
    _acceleration(figure.add_subplot(grid[2, :]), palette, time, record, missile)

    heading = title if title is not None else "Engagement"
    figure.suptitle(
        heading, color=palette.text, fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.965
    )
    if intercept is not None:
        figure.text(
            0.055,
            0.925,
            f"{'Hit' if intercept.hit else 'Miss'} · {intercept.miss_distance:.2f} m at "
            f"t = {intercept.time:.2f} s · closing at {intercept.closing_speed:.0f} m/s",
            color=palette.muted,
            fontsize=10.5,
            ha="left",
        )
    return figure


def _plan_view(
    ax: Axes,
    theme: Theme,
    missile_pos: Vector,
    target_pos: Vector,
    intercept: Intercept | None,
) -> None:
    ax.plot(missile_pos[:, 0], missile_pos[:, 1], color=theme.missile, linewidth=2.0)
    ax.plot(target_pos[:, 0], target_pos[:, 1], color=theme.target, linewidth=2.0)

    # Open markers for the launch points; identity is also carried by the label.
    ax.plot(
        missile_pos[0, 0],
        missile_pos[0, 1],
        "o",
        color=theme.missile,
        markersize=8,
        markerfacecolor=theme.surface,
        markeredgewidth=2,
    )
    ax.plot(
        target_pos[0, 0],
        target_pos[0, 1],
        "o",
        color=theme.target,
        markersize=8,
        markerfacecolor=theme.surface,
        markeredgewidth=2,
    )
    ax.annotate(
        "Missile",
        (missile_pos[0, 0], missile_pos[0, 1]),
        textcoords="offset points",
        xytext=(10, -12),
        color=theme.text,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.annotate(
        "Target",
        (target_pos[0, 0], target_pos[0, 1]),
        textcoords="offset points",
        xytext=(10, 8),
        color=theme.text,
        fontsize=9.5,
        fontweight="bold",
    )

    if intercept is not None:
        ax.plot(
            missile_pos[-1, 0],
            missile_pos[-1, 1],
            "x",
            color=theme.text,
            markersize=11,
            markeredgewidth=2.2,
        )
        ax.annotate(
            f"{intercept.miss_distance:.1f} m",
            (missile_pos[-1, 0], missile_pos[-1, 1]),
            textcoords="offset points",
            xytext=(12, 6),
            color=theme.text,
            fontsize=9.5,
        )

    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, theme, "East (m)", "North (m)", "Plan view")


def _side_view(ax: Axes, theme: Theme, missile_pos: Vector, target_pos: Vector) -> None:
    origin = missile_pos[0, :2]
    missile_range = np.linalg.norm(missile_pos[:, :2] - origin, axis=1)
    target_range = np.linalg.norm(target_pos[:, :2] - origin, axis=1)

    ax.plot(missile_range, missile_pos[:, 2], color=theme.missile, linewidth=2.0, label="Missile")
    ax.plot(target_range, target_pos[:, 2], color=theme.target, linewidth=2.0, label="Target")
    ax.axhline(0.0, color=theme.limit, linewidth=1.2, linestyle=":")

    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(theme.text)

    _style(ax, theme, "Ground distance from launch (m)", "Altitude (m)", "Side view")


def _separation(
    ax: Axes,
    theme: Theme,
    time: Vector,
    record: Recorder,
    missile: str,
    target: str,
    intercept: Intercept | None,
) -> None:
    separation = record.separation(target, missile)
    # Guard the log axis: the final sample can legitimately be sub-millimetre.
    ax.semilogy(time, np.maximum(separation, 1e-3), color=theme.text, linewidth=2.0)

    if intercept is not None:
        ax.axhline(intercept.miss_distance, color=theme.missile, linewidth=1.5, linestyle="--")
        ax.annotate(
            f"closest approach {intercept.miss_distance:.2f} m",
            (time[0], intercept.miss_distance),
            textcoords="offset points",
            xytext=(6, 6),
            color=theme.text,
            fontsize=9,
        )

    _style(ax, theme, "Time (s)", "Separation (m)", "Range to target")


def _speed(
    ax: Axes, theme: Theme, time: np.ndarray, record: Recorder, missile: str, target: str
) -> None:
    ax.plot(
        time,
        record.speed(missile),
        color=theme.missile,
        linewidth=2.0,
        label="Missile",
    )
    ax.plot(
        time,
        record.speed(target),
        color=theme.target,
        linewidth=2.0,
        label="Target",
    )

    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(theme.text)

    _style(ax, theme, "Time (s)", "Speed (m/s)", "Speed")


def _acceleration(ax: Axes, theme: Theme, time: np.ndarray, record: Recorder, missile: str) -> None:
    commanded = record.commanded(missile) / _G
    achieved = record.achieved(missile) / _G
    limit = record.limit(missile) / _G

    ax.plot(time, commanded, color=theme.missile, linewidth=1.6, linestyle="--", label="Commanded")
    ax.plot(time, achieved, color=theme.missile, linewidth=2.4, label="Achieved")
    ax.plot(time, limit, color=theme.limit, linewidth=1.6, linestyle=":", label="Airframe limit")

    # Commanded acceleration diverges towards infinity at closest approach, so a
    # linear axis scaled to the peak would flatten the whole flight into a line.
    ceiling = float(np.percentile(limit[limit > 0.0], 95)) if np.any(limit > 0.0) else 1.0
    ax.set_ylim(0.0, max(ceiling * 1.6, float(achieved.max()) * 1.3, 1.0))

    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    for text in legend.get_texts():
        text.set_color(theme.text)

    _style(
        ax,
        theme,
        "Time (s)",
        "Lateral acceleration (g)",
        "Guidance demand vs what the airframe delivered",
    )


def save_engagement(
    result: RunResult,
    path: str | Path,
    *,
    missile: str = "missile",
    target: str = "target",
    intercept: Intercept | None = None,
    title: str | None = None,
    theme: str = "light",
) -> Path:
    """Render the diagnostic figure and write it to ``path``.

    Closes the figure afterwards, so a loop over many scenarios does not leak
    figures and trip matplotlib's open-figure warning.
    """
    import matplotlib.pyplot as plt

    figure = plot_engagement(
        result, missile=missile, target=target, intercept=intercept, title=title, theme=theme
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return destination


@dataclass(frozen=True)
class LawRun:
    """One guidance law's result, ready to be compared against another's."""

    label: str
    result: RunResult
    intercept: Intercept | None


def plot_comparison(
    runs: Sequence[LawRun],
    *,
    missile: str = "missile",
    target: str = "target",
    title: str | None = None,
    theme: str = "light",
) -> Figure:
    """Compare several guidance laws flown on the same scenario.

    Three panels: the plan view, which shows the *shape* of each law's solution
    and is the panel that actually explains the difference; separation against
    time; and achieved lateral acceleration, which answers whether a better
    intercept was bought with more effort or less.

    Raises:
        ValueError: if there are more laws than validated colour slots. The
            palette is fixed and checked, not generated — see :class:`Theme`.
    """
    import matplotlib.pyplot as plt

    if theme not in THEMES:
        msg = f"theme must be one of {sorted(THEMES)}, got {theme!r}"
        raise ValueError(msg)
    palette = THEMES[theme]

    if not runs:
        msg = "nothing to compare"
        raise ValueError(msg)
    if len(runs) > len(palette.laws):
        msg = (
            f"{len(runs)} laws but only {len(palette.laws)} validated colour slots; "
            "add a slot to Theme.laws and re-run the palette validator"
        )
        raise ValueError(msg)

    figure = plt.figure(figsize=(13.0, 8.5), dpi=110, facecolor=palette.surface)
    grid = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.22, top=0.87, bottom=0.08)

    # The plan view takes the full-height left column rather than a wide strip
    # across the top. An engagement is typically much longer downrange than it
    # is wide, so a tall box fits the data under equal aspect without either
    # distorting the geometry or leaving most of the panel empty.
    plan = figure.add_subplot(grid[:, 0])
    separation = figure.add_subplot(grid[0, 1])
    acceleration = figure.add_subplot(grid[1, 1])

    # The target's path is the same in every run, so draw it once.
    reference = runs[0].result.recorder.position(target)
    plan.plot(reference[:, 0], reference[:, 1], color=palette.target, linewidth=2.0)
    plan.annotate(
        "Target",
        (reference[0, 0], reference[0, 1]),
        textcoords="offset points",
        xytext=(10, 8),
        color=palette.text,
        fontsize=9.5,
        fontweight="bold",
    )

    for index, run_ in enumerate(runs):
        colour = palette.laws[index]
        record = run_.result.recorder
        path = record.position(missile)
        miss = "" if run_.intercept is None else f" · {run_.intercept.miss_distance:.2f} m"

        plan.plot(path[:, 0], path[:, 1], color=colour, linewidth=2.2, label=run_.label + miss)
        plan.plot(
            path[-1, 0],
            path[-1, 1],
            "x",
            color=colour,
            markersize=11,
            markeredgewidth=2.2,
        )

        separation.semilogy(
            record.time,
            np.maximum(record.separation(target, missile), 1e-3),
            color=colour,
            linewidth=2.0,
            label=run_.label,
        )
        acceleration.plot(
            record.time,
            record.achieved(missile) / _G,
            color=colour,
            linewidth=2.0,
            label=run_.label,
        )

    plan.plot(
        reference[0, 0],
        reference[0, 1],
        "o",
        color=palette.target,
        markersize=8,
        markerfacecolor=palette.surface,
        markeredgewidth=2,
    )
    # "box" rather than "datalim": this panel spans the full figure width, and
    # padding the data limits to fill it would stretch a 3 km engagement across
    # a 17 km axis. Shrinking the axes box instead keeps the scale honest and
    # the tick labels meaningful.
    plan.set_aspect("equal", adjustable="box")

    for axis, xlabel, ylabel, heading, corner in (
        (plan, "East (m)", "North (m)", "Plan view — the shape of each solution", "lower right"),
        (separation, "Time (s)", "Separation (m)", "Range to target", "best"),
        (
            acceleration,
            "Time (s)",
            "Lateral acceleration (g)",
            "Acceleration actually used",
            "best",
        ),
    ):
        legend = axis.legend(frameon=False, fontsize=9, loc=corner)
        for text in legend.get_texts():
            text.set_color(palette.text)
        _style(axis, palette, xlabel, ylabel, heading)

    figure.suptitle(
        title if title is not None else "Guidance law comparison",
        color=palette.text,
        fontsize=15,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=0.955,
    )
    summary = "   ".join(
        f"{r.label}: {'—' if r.intercept is None else f'{r.intercept.miss_distance:.2f} m'}"
        for r in runs
    )
    figure.text(0.055, 0.915, summary, color=palette.muted, fontsize=10.5, ha="left")
    return figure


def save_comparison(
    runs: Sequence[LawRun],
    path: str | Path,
    *,
    missile: str = "missile",
    target: str = "target",
    title: str | None = None,
    theme: str = "light",
) -> Path:
    """Render the comparison figure and write it to ``path``."""
    import matplotlib.pyplot as plt

    figure = plot_comparison(runs, missile=missile, target=target, title=title, theme=theme)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_noise_sweep(
    angle_sigmas_mrad: Sequence[float],
    misses: Vector,
    *,
    lethal_radius: float = 5.0,
    baseline: float | None = None,
    title: str | None = None,
    theme: str = "light",
) -> Figure:
    """Miss distance against seeker angle noise, on log-log axes.

    Args:
        angle_sigmas_mrad: The noise levels swept, in milliradians.
        misses: Shape ``(len(angle_sigmas_mrad), seeds)``. Every seed is drawn
            as a band around the median, because a single seed of a stochastic
            process is an anecdote.
        lethal_radius: Drawn as a threshold — above this line the engagement
            failed, whatever the trend looks like.
        baseline: Miss distance on perfect information, if known. Drawn as the
            floor the sweep is degrading away from.

    Log-log because both axes span orders of magnitude, and because a power law
    plots as a straight line there — the slope then reads directly as "miss
    distance grows as roughly the square of angle noise", which is the finding.
    """
    import matplotlib.pyplot as plt

    if theme not in THEMES:
        msg = f"theme must be one of {sorted(THEMES)}, got {theme!r}"
        raise ValueError(msg)
    palette = THEMES[theme]

    sigmas = np.asarray(angle_sigmas_mrad, dtype=np.float64)
    data = np.atleast_2d(np.asarray(misses, dtype=np.float64))
    if data.shape[0] != sigmas.size:
        msg = f"expected {sigmas.size} rows of misses, got {data.shape[0]}"
        raise ValueError(msg)

    median = np.median(data, axis=1)

    figure = plt.figure(figsize=(9.0, 6.2), dpi=110, facecolor=palette.surface)
    ax = figure.add_subplot(111)

    ax.fill_between(
        sigmas,
        data.min(axis=1),
        data.max(axis=1),
        color=palette.missile,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(sigmas, median, color=palette.missile, linewidth=2.4, marker="o", markersize=7)

    ax.axhline(lethal_radius, color=palette.limit, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"lethal radius {lethal_radius:.0f} m — above this line the engagement failed",
        (sigmas[0], lethal_radius),
        textcoords="offset points",
        xytext=(4, 6),
        color=palette.text,
        fontsize=9,
    )

    if baseline is not None and baseline > 0.0:
        ax.axhline(baseline, color=palette.target, linewidth=1.4, linestyle=":")
        ax.annotate(
            f"perfect information: {baseline:.2f} m",
            (sigmas[0], baseline),
            textcoords="offset points",
            xytext=(4, 6),
            color=palette.text,
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # The figure heading names the chart, so the axes title would only repeat it.
    _style(ax, palette, "Seeker angle noise, one sigma (mrad)", "Miss distance (m)", "")
    figure.suptitle(
        title if title is not None else "Seeker noise sweep",
        color=palette.text,
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.02,
        0.905,
        "Band spans every random seed; the line is the median.",
        color=palette.muted,
        fontsize=10,
        ha="left",
    )
    figure.subplots_adjust(top=0.855, bottom=0.1, left=0.1, right=0.97)
    return figure


def save_noise_sweep(
    angle_sigmas_mrad: Sequence[float],
    misses: Vector,
    path: str | Path,
    *,
    lethal_radius: float = 5.0,
    baseline: float | None = None,
    title: str | None = None,
    theme: str = "light",
) -> Path:
    """Render the noise sweep and write it to ``path``."""
    import matplotlib.pyplot as plt

    figure = plot_noise_sweep(
        angle_sigmas_mrad,
        misses,
        lethal_radius=lethal_radius,
        baseline=baseline,
        title=title,
        theme=theme,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return destination
