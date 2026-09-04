"""A real-time 3D window you can orbit while the engagement plays.

The other half of Phase 6, and the half that cannot be tested. A recording is a
file and a file can be asserted about; a window is something a person looks at.
So this module is kept as thin as it can be — it places objects where
:class:`~interceptor.viz.scene.Storyboard` says and advances a clock — and every
decision with arithmetic in it lives in the scene model, which is tested.

**Nothing imports this module.** Not the simulation core, not
:mod:`interceptor.viz.plots`, not the CLI at module scope. VPython starts a web
server and opens a browser tab as a side effect of being imported, which is
splendid when you asked for a viewer and intolerable when you asked for a
headless Monte Carlo run. The import is therefore inside the function, and the
absence of VPython is reported as a suggestion rather than a traceback.

**Why VPython and not a matplotlib 3D window.** matplotlib's 3D is a projection
of 2D primitives with no depth buffer: lines pass through spheres rather than
behind them, and rotating a few thousand trail points is slow enough to be
unpleasant. VPython is a real scene graph rendered by WebGL, it orbits smoothly
with a mouse, and it runs in a browser tab — which on this project's original
Windows machine is a positive advantage, since that machine's Tcl/Tk is broken
and every window toolkit that depends on it fails.
"""

from __future__ import annotations

import time
from typing import Any

from interceptor.viz.scene import Storyboard

__all__ = ["view"]

MISSILE_COLOUR = (0.91, 0.34, 0.13)
TARGET_COLOUR = (0.13, 0.51, 0.85)
"""Matching the plot palette. VPython wants 0-1 floats, not hex."""

TRAIL_RADIUS_FRACTION = 0.0016
"""Trail thickness as a fraction of the scene's width, so a 6 km engagement and
a 600 m one both come out looking sensible rather than one being a hairline and
the other a drainpipe."""


def _require_vpython() -> Any:
    """Import VPython, or explain how to get it.

    Deliberately not a module-level import: see the note at the top about what
    VPython does on import.
    """
    try:
        import vpython
    except ImportError as error:  # pragma: no cover - depends on the environment
        msg = (
            "the live viewer needs VPython, which is an optional extra.\n"
            "  pip install -e '.[live]'      (or: pip install vpython)\n"
            "Everything else in interceptor.viz works without it, and\n"
            "'interceptor record' writes an animation with no extra install."
        )
        raise RuntimeError(msg) from error
    return vpython


def view(
    storyboard: Storyboard,
    *,
    speed: float = 1.0,
    loop: bool = False,
    width: int = 1000,
    height: int = 640,
) -> None:
    """Play an engagement in an interactive 3D window.

    Args:
        storyboard: What to draw, from
            :func:`~interceptor.viz.scene.storyboard_from`.
        speed: Playback rate relative to the storyboard's own frame rate. The
            storyboard may already be slowed; this multiplies that.
        loop: Replay from the start when it finishes rather than stopping.
        width: Canvas size in pixels.
        height: Canvas size in pixels.

    Blocks until the animation finishes. The window stays open afterwards —
    VPython keeps serving it — so the final geometry can be inspected and
    orbited, which is usually the point.
    """
    if not storyboard.frames:
        msg = "storyboard has no frames"
        raise ValueError(msg)
    if speed <= 0.0:
        msg = f"speed must be positive, got {speed}"
        raise ValueError(msg)

    vp = _require_vpython()

    lower, upper = storyboard.bounds
    centre = (lower + upper) / 2.0
    extent = float(max(upper - lower))
    trail_radius = max(extent * TRAIL_RADIUS_FRACTION, 0.5)

    scene = vp.canvas(
        title=storyboard.title or "Intercept",
        width=width,
        height=height,
        background=vp.color.white * 0.97,
        center=vp.vector(*centre),
        range=extent * 0.6,
        forward=vp.vector(-0.6, -1.0, -0.35),
    )
    scene.caption = (
        "\nDrag to orbit, scroll to zoom, shift-drag to pan.\n"
        "Orange is the missile, blue the target; the dotted line is the sightline.\n"
    )

    first = storyboard.frames[0]
    missile = vp.sphere(
        pos=vp.vector(*first.missile_position),
        radius=trail_radius * 4.0,
        color=vp.vector(*MISSILE_COLOUR),
        make_trail=True,
        trail_radius=trail_radius,
        retain=len(storyboard.frames),
    )
    target = vp.sphere(
        pos=vp.vector(*first.target_position),
        radius=trail_radius * 4.0,
        color=vp.vector(*TARGET_COLOUR),
        make_trail=True,
        trail_radius=trail_radius,
        retain=len(storyboard.frames),
    )
    sightline = vp.curve(
        color=vp.color.gray(0.55),
        radius=trail_radius * 0.35,
    )
    readout = vp.label(
        pos=vp.vector(*centre),
        pixel_pos=False,
        xoffset=-width * 0.42,
        yoffset=height * 0.36,
        text="",
        align="left",
        box=False,
        opacity=0,
        color=vp.color.black,
        height=13,
        font="monospace",
    )

    def draw(index: int) -> None:
        frame = storyboard.frames[index]
        missile.pos = vp.vector(*frame.missile_position)
        target.pos = vp.vector(*frame.target_position)
        sightline.clear()
        sightline.append(missile.pos)
        sightline.append(target.pos)
        readout.text = (
            f"t       {frame.time:6.2f} s\n"
            f"range   {frame.range:6.0f} m\n"
            f"closing {frame.closing_speed:6.0f} m/s\n"
            f"LOS     {frame.los_rate:6.3f} rad/s\n"
            f"used    {frame.achieved_g:6.1f} g\n"
            f"speed   {frame.speed:6.0f} m/s"
        )

    rate = storyboard.fps * speed

    while True:
        missile.clear_trail()
        target.clear_trail()
        for index in range(len(storyboard.frames)):
            # vp.rate() both throttles the loop to real time and gives the
            # browser its chance to render and to handle the mouse. Replacing it
            # with time.sleep produces a window that ignores you.
            vp.rate(rate)
            draw(index)

        if storyboard.intercept is not None:
            outcome = "HIT" if storyboard.intercept.hit else "miss"
            readout.text += f"\n\n{outcome} by {storyboard.intercept.miss_distance:.2f} m"

        if not loop:
            break
        time.sleep(1.0)
