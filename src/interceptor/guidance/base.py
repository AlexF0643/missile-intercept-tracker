"""The guidance law interface.

A law takes a :class:`~interceptor.sensing.track.Track` and the missile's own
state, and returns a commanded acceleration in the world frame. It does not
know where the track came from, cannot reach into the world, and holds no
opinion about the airframe's limits — clamping happens downstream in the
autopilot, so that "what was asked for" and "what was achievable" stay separate
and both can be plotted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from interceptor.core.state import EntityState, Vector
from interceptor.sensing.track import Track

__all__ = ["GuidanceLaw", "perpendicular_component"]


def perpendicular_component(vector: Vector, reference: Vector) -> Vector:
    """The part of ``vector`` at right angles to ``reference``.

    Guidance commands are lateral by definition — a component along the
    velocity would be a throttle setting, which a missile does not have. Every
    law in this package projects its command through here.
    """
    denominator = float(np.dot(reference, reference))
    if denominator < 1e-12:
        return np.asarray(vector, dtype=np.float64)
    along = (float(np.dot(vector, reference)) / denominator) * reference
    return np.asarray(vector - along, dtype=np.float64)


class GuidanceLaw(ABC):
    """Turns an estimated track into a commanded lateral acceleration."""

    @abstractmethod
    def command(self, track: Track, missile: EntityState) -> Vector:
        """Commanded acceleration in the world frame, m/s^2.

        Must return zeros when ``track.valid`` is ``False`` or the geometry is
        degenerate. Coasting straight is the correct response to knowing
        nothing; guessing is not.
        """

    @property
    def name(self) -> str:
        """Human-readable name, used in plot legends and printed results."""
        return type(self).__name__
