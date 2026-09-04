"""Render an engagement to an animated file.

The half of Phase 6 that can be checked by a machine. A live 3D window is
something a person looks at; a recording is a file, and a file can be asserted
about — it exists, it has the right number of frames, it is not two kilobytes of
blank canvas. So the recorder carries the phase's exit criterion, and the live
viewer in :mod:`interceptor.viz.live` is a second renderer over the same
:class:`~interceptor.viz.scene.Storyboard` rather than a separate implementation
that might disagree with it.

Headless by construction, like the rest of :mod:`interceptor.viz`: a bare
``Figure`` with an Agg canvas, never ``pyplot``. Writing an animation on a build
server with no display has to work, and on a desktop it must not steal focus by
opening a window nobody asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from interceptor.viz.plots import THEMES, Theme, _new_figure
from interceptor.viz.scene import Storyboard

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["render_flight", "save_flight"]

TRAIL_WIDTH = 2.0
ORBIT_DEGREES = 24.0
"""How far the camera drifts over the whole animation when ``orbit`` is on.

Parallax is what makes a projected 3D path read as depth rather than as a
squiggle, so it is tempting to leave on. It is off by default anyway, because it
triples the file: a moving camera changes every pixel of the background on every
frame, and a GIF stores frames as differences from the one before. The same
animation is 1.2 MB with a fixed camera and 3.5 MB with a drifting one, for the
same 223 frames.

Depth perception is what the live viewer is for — there you can orbit the scene
yourself, with a mouse, which beats any preset drift.
"""


VERTICAL_FRACTION = 0.34
"""Height of the plot box relative to its horizontal side.

An air-to-air engagement is nearly flat: 6 km across and perhaps 150 m of
altitude change. Drawn to a true cube it is a thin smear through six kilometres
of empty sky, and the vertical manoeuvre — the thing a 3D view exists to show —
disappears entirely. So the vertical is exaggerated, which is a distortion, and
the honest response to a necessary distortion is to state it: the axis label
carries the factor.
"""


def _extent(storyboard: Storyboard) -> tuple[np.ndarray, float, float, float]:
    """Axis limits, and how much the vertical has been stretched.

    The two horizontal axes share a scale. That one is not negotiable — it is
    what makes a crossing engagement look like a crossing rather than a head-on,
    and it is the difference between a picture of the engagement and a picture
    of the axis limits. The vertical gets its own scale, and the caller is told
    the ratio so it can be written on the figure.
    """
    lower, upper = storyboard.bounds
    centre = (lower + upper) / 2.0
    span = upper - lower

    horizontal = max(float(np.max(span[:2])) * 1.12, 1.0)
    vertical = max(float(span[2]) * 1.5, horizontal * 0.02, 50.0)
    exaggeration = VERTICAL_FRACTION * horizontal / vertical
    return centre, horizontal, vertical, exaggeration


def _readout(frame: Any, intercept: Any, *, final: bool = False) -> str:
    """The live numbers, plus the outcome once it has happened.

    Range and closing speed say where things are; the line-of-sight rate says
    whether the intercept is working. A sightline that holds a steady bearing
    while the range falls is a collision, which is the whole idea proportional
    navigation is built on, so it is the number worth watching.

    That rate diverges in the last moments, and the field is wide enough to show
    it doing so. This is not a display bug: ``Omega = (r x v) / (r . r)`` has the
    range squared underneath, so as the range goes to nothing the sightline
    sweeps arbitrarily fast. It is the reason terminal guidance saturates however
    much airframe you give it, and watching the number run away is a better
    explanation of that than a paragraph.
    """
    text = (
        f"t       {frame.time:7.2f} s\n"
        f"range   {frame.range:7.0f} m\n"
        f"closing {frame.closing_speed:7.0f} m/s\n"
        f"LOS     {frame.los_rate:7.3f} rad/s\n"
        f"used    {frame.achieved_g:7.1f} g\n"
        f"speed   {frame.speed:7.0f} m/s"
    )
    # `final` as well as the time comparison: closest approach is solved for by
    # interpolation between two steps, so it usually falls a few milliseconds
    # after the last recorded sample and no frame would ever satisfy the test.
    if intercept is not None and (final or frame.time >= intercept.time):
        outcome = "HIT" if intercept.hit else "miss"
        text += f"\n\n{outcome} by {intercept.miss_distance:.2f} m"
    return text


def render_flight(
    storyboard: Storyboard,
    *,
    theme: str = "light",
    orbit: bool = False,
) -> tuple[Figure, Callable[[int], None]]:
    """Build the figure, and a function that draws any frame onto it.

    Returns a plain callable rather than a
    :class:`~matplotlib.animation.FuncAnimation`. An animation object owns a
    timer and complains when it is collected without having been rendered, so
    creating one here would mean anybody who merely wanted to look at a single
    frame got a warning for their trouble. :func:`save_flight` builds one when
    there is actually a file to write, which is the only place it is needed.
    """
    if theme not in THEMES:
        msg = f"theme must be one of {sorted(THEMES)}, got {theme!r}"
        raise ValueError(msg)
    if not storyboard.frames:
        msg = "storyboard has no frames"
        raise ValueError(msg)

    palette: Theme = THEMES[theme]
    figure = _new_figure(6.8, 3.9, palette.surface)
    ax = figure.add_subplot(projection="3d")
    ax.set_facecolor(palette.surface)

    centre, horizontal, vertical, exaggeration = _extent(storyboard)
    ax.set_xlim3d(centre[0] - horizontal / 2, centre[0] + horizontal / 2)
    ax.set_ylim3d(centre[1] - horizontal / 2, centre[1] + horizontal / 2)
    ax.set_zlim3d(centre[2] - vertical / 2, centre[2] + vertical / 2)
    ax.set_box_aspect((1.0, 1.0, VERTICAL_FRACTION))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(palette.muted)
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = palette.grid
    ax.set_xlabel("East (m)", fontsize=8, color=palette.muted)
    ax.set_ylabel("North (m)", fontsize=8, color=palette.muted)
    ax.set_zlabel(f"Up (m), scale x{exaggeration:.0f}", fontsize=8, color=palette.muted)
    ax.tick_params(colors=palette.muted, labelsize=7)
    # The vertical spans a couple of hundred metres against kilometres
    # horizontally, so its default tick density is several times the others.
    from matplotlib.ticker import MaxNLocator

    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    (missile_trail,) = ax.plot([], [], [], color=palette.missile, linewidth=TRAIL_WIDTH)
    (target_trail,) = ax.plot([], [], [], color=palette.target, linewidth=TRAIL_WIDTH)
    (missile_dot,) = ax.plot([], [], [], "o", color=palette.missile, markersize=6)
    (target_dot,) = ax.plot([], [], [], "o", color=palette.target, markersize=6)
    (sightline,) = ax.plot([], [], [], color=palette.muted, linewidth=0.9, linestyle=":", alpha=0.9)

    # Title, readout and key live in *figure* coordinates rather than axes
    # coordinates. A 3D axes reports a bounding box far larger than the box it
    # actually draws in, so anything anchored to it drifts as the camera turns
    # and collides with the title. Pinning them to the figure gives a stable
    # band across the top and lets the 3D panel fill everything beneath it.
    if storyboard.title:
        figure.text(
            0.02,
            0.965,
            storyboard.title,
            color=palette.text,
            fontsize=11.5 if len(storyboard.title) > 46 else 12.5,
            fontweight="bold",
            ha="left",
            va="top",
        )

    readout = figure.text(
        0.02,
        0.885,
        "",
        color=palette.text,
        fontsize=8.5,
        family="monospace",
        ha="left",
        va="top",
    )

    # Two text objects rather than one two-line label, because the whole point
    # of the key is that each word is the colour of the thing it names.
    for offset, (word, colour) in enumerate(
        (("missile", palette.missile), ("target", palette.target))
    ):
        figure.text(
            0.98,
            0.885 - 0.055 * offset,
            word,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
            color=colour,
        )

    azimuth_start = -62.0
    total = max(len(storyboard.frames) - 1, 1)

    def draw(index: int) -> None:
        frame = storyboard.frames[index]
        missile_trail.set_data_3d(
            frame.missile_trail[:, 0], frame.missile_trail[:, 1], frame.missile_trail[:, 2]
        )
        target_trail.set_data_3d(
            frame.target_trail[:, 0], frame.target_trail[:, 1], frame.target_trail[:, 2]
        )
        missile_dot.set_data_3d(*[[c] for c in frame.missile_position])
        target_dot.set_data_3d(*[[c] for c in frame.target_position])
        sightline.set_data_3d(
            [frame.missile_position[0], frame.target_position[0]],
            [frame.missile_position[1], frame.target_position[1]],
            [frame.missile_position[2], frame.target_position[2]],
        )
        readout.set_text(_readout(frame, storyboard.intercept, final=index == total))
        if orbit:
            ax.view_init(elev=20.0, azim=azimuth_start + ORBIT_DEGREES * index / total)

    figure.subplots_adjust(left=0.0, right=1.0, bottom=-0.10, top=0.90)
    draw(0)
    return figure, draw


def save_flight(
    storyboard: Storyboard,
    path: str | Path,
    *,
    theme: str = "light",
    orbit: bool = False,
    dpi: int = 85,
) -> Path:
    """Write the animation to ``path``.

    The suffix chooses the writer: ``.gif`` uses Pillow, which ships with
    matplotlib and therefore always works; ``.mp4`` uses ffmpeg, which does not
    ship with anything and is checked for rather than assumed.
    """
    destination = Path(path)

    # Choose the writer before building anything. A FuncAnimation that is
    # created and then never rendered warns from its destructor, so discovering
    # a missing ffmpeg *after* constructing one turns a clear error message into
    # a clear error message plus an unraisable warning attributed to whichever
    # unrelated test the garbage collector happened to interrupt. Ask the
    # question first and there is nothing to collect.
    if destination.suffix.lower() == ".mp4":
        from matplotlib.animation import FFMpegWriter

        if not FFMpegWriter.isAvailable():
            msg = (
                "writing .mp4 needs ffmpeg on PATH, which was not found. "
                "Use a .gif filename instead — Pillow ships with matplotlib."
            )
            raise RuntimeError(msg)
        writer: Any = FFMpegWriter(fps=round(storyboard.fps), bitrate=2400)
    else:
        from matplotlib.animation import PillowWriter

        writer = PillowWriter(fps=round(storyboard.fps))

    destination.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.animation import FuncAnimation

    figure, draw = render_flight(storyboard, theme=theme, orbit=orbit)
    animation = FuncAnimation(
        figure,
        # The stubs demand a function returning artists, which is only true
        # under blit=True. Without blitting matplotlib ignores the return value
        # and redraws the lot, which is what a rotating 3D axes needs anyway.
        draw,  # type: ignore[arg-type]
        frames=len(storyboard.frames),
        interval=1000.0 / storyboard.fps,
        blit=False,
    )

    animation.save(
        str(destination),
        writer=writer,
        dpi=dpi,
        savefig_kwargs={"facecolor": figure.get_facecolor()},
    )
    return destination
