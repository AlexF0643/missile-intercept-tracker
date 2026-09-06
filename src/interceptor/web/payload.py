"""A flown engagement, as JSON the browser can draw.

This is the arithmetic half of the browser app, kept away from the HTTP half so
that it can be tested without a socket. It takes the same
:class:`~interceptor.viz.scene.Storyboard` the GIF recorder and the VPython
window consume and turns it into a dictionary — no request, no response, no
server.

**The shape is chosen to keep the payload linear.** A :class:`Frame` carries
cumulative trails, which is exactly right for a renderer holding one frame at a
time and quadratic for one sending every frame down a wire: four hundred frames
of a growing trail is eighty thousand points for a path with four hundred in it.
So the two trajectories are sent once, in full, and each frame carries the
*index* into them. The browser slices; the wire does not repeat itself.

Coordinates are rounded to a centimetre. The simulation integrates in double
precision and should keep doing so, but seventeen significant figures of a
position that is a metre wide is ten times the bytes for none of the meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interceptor.sim.intercept import Intercept
from interceptor.viz.scene import Storyboard

__all__ = ["storyboard_payload"]

#: Metres. A centimetre is far finer than anything drawn at screen resolution,
#: and far coarser than the noise floor of the simulation, so nothing visible
#: and nothing meaningful is lost.
_POSITION_DECIMALS = 2


def _path(points: np.ndarray) -> list[list[float]]:
    """A trajectory as plain nested lists, rounded."""
    return [[round(float(v), _POSITION_DECIMALS) for v in point] for point in points]


def _intercept(storyboard: Storyboard) -> dict[str, Any] | None:
    """The closest approach, with somewhere to draw a marker.

    Closest approach is solved for by interpolating between two integration
    steps, so it has a time but no recorded position of its own. The nearest
    frame is within one frame interval of it and is what both other renderers
    mark, so it is what this marks too.
    """
    intercept: Intercept | None = storyboard.intercept
    if intercept is None:
        return None
    where = storyboard.at(intercept.time).missile_position
    return {
        "time": round(intercept.time, 4),
        "missDistance": round(intercept.miss_distance, 3),
        "closingSpeed": round(intercept.closing_speed, 2),
        "hit": bool(intercept.hit),
        "position": [round(float(v), _POSITION_DECIMALS) for v in where],
    }


def storyboard_payload(
    storyboard: Storyboard,
    *,
    missile_path: np.ndarray,
    target_path: np.ndarray,
    subtitle: str = "",
) -> dict[str, Any]:
    """Everything the browser needs to draw one engagement.

    Args:
        storyboard: The resampled frames, from
            :func:`~interceptor.viz.scene.storyboard_from`.
        missile_path: Every recorded missile position, shape ``(n, 3)``. The
            frames index into this rather than carrying their own copies.
        target_path: The same for the target.
        subtitle: What was flown — law, seeker, estimator.

    The frame keys are short because there are hundreds of them and the names
    are repeated in every one; the top-level keys are not, because there is one
    of each and clarity is free.
    """
    lower, upper = storyboard.bounds
    return {
        "title": storyboard.title,
        "subtitle": subtitle,
        "fps": storyboard.fps,
        "duration": round(storyboard.duration, 4),
        "bounds": {
            "lower": [round(float(v), _POSITION_DECIMALS) for v in lower],
            "upper": [round(float(v), _POSITION_DECIMALS) for v in upper],
        },
        "intercept": _intercept(storyboard),
        "paths": {"missile": _path(missile_path), "target": _path(target_path)},
        "frames": [
            {
                "t": round(frame.time, 4),
                # Index into `paths`; the trail is everything up to and
                # including it.
                "i": frame.sample,
                "range": round(frame.range, 2),
                "closing": round(frame.closing_speed, 2),
                "los": round(frame.los_rate, 5),
                "cmd": round(frame.commanded_g, 3),
                "used": round(frame.achieved_g, 3),
                "speed": round(frame.speed, 2),
            }
            for frame in storyboard.frames
        ],
    }
