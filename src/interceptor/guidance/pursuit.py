"""Pure pursuit — the baseline, and a deliberately poor one.

Pure pursuit points the missile at where the target *is*. That sounds
reasonable and is the intuitive thing a person or a dog does, but it has a
specific and fatal weakness: against a target that is crossing rather than
running away, the missile is always steering at a point the target has already
left. It swings in behind and ends up in a tail chase, arriving late, slow, and
needing enormous acceleration in the last second to close the final gap.

It is here to be beaten. Phase 3 adds proportional navigation, which steers at
where the target is *going*, and the comparison between the two on the same
scenario is the clearest possible argument for why PN is what real systems use.
"""

from __future__ import annotations

import numpy as np

from interceptor.core.state import EntityState, Vector
from interceptor.guidance.base import GuidanceLaw, perpendicular_component
from interceptor.sensing.track import Track

__all__ = ["PurePursuit"]

_EPS = 1e-9


class PurePursuit(GuidanceLaw):
    """Turn towards the line of sight at a rate proportional to the angle off it.

    The command is ``a = K * V * sin(theta) * n``, where ``theta`` is the angle
    between the velocity and the line of sight, and ``n`` is the unit vector
    perpendicular to the velocity in the plane containing both. Writing it with
    ``sin(theta)`` rather than ``theta`` keeps it a pure vector expression with
    no arctangent and no quadrant handling, and the two agree for small angles
    anyway.

    Args:
        gain: Turn-rate gain in units of 1/s. Around 4 gives a brisk response
            without oscillating. Raise it and the missile turns harder onto the
            sightline; it still cannot fix the fundamental lag.
    """

    def __init__(self, gain: float = 4.0) -> None:
        if gain <= 0.0:
            msg = f"gain must be positive, got {gain}"
            raise ValueError(msg)
        self.gain = gain

    def command(self, track: Track, missile: EntityState) -> Vector:
        if not track.valid:
            return np.zeros(3, dtype=np.float64)

        speed = missile.speed
        if speed < _EPS or track.range < _EPS:
            return np.zeros(3, dtype=np.float64)

        # Component of the sightline at right angles to where we are pointing.
        # Its length is sin(theta) because the sightline is a unit vector.
        lateral = perpendicular_component(track.line_of_sight, missile.vel)
        magnitude = float(np.linalg.norm(lateral))
        if magnitude < _EPS:
            return np.zeros(3, dtype=np.float64)  # already pointing straight at it

        direction = lateral / magnitude
        return np.asarray(self.gain * speed * magnitude * direction, dtype=np.float64)
