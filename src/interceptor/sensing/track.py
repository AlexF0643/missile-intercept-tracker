"""The engagement geometry as the guidance law sees it.

A :class:`Track` is the *only* thing a guidance law is ever given. In this phase
it is filled in from truth, so the numbers are perfect; from Phase 4 the same
object arrives from a noisy seeker and an estimator, and not one line of the
guidance code has to change. That is the whole reason this type exists rather
than passing the two entities around.

The derived quantities live here too — range, closing velocity and the
line-of-sight rotation vector — because every guidance law needs them and there
is exactly one correct way to compute each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from interceptor.core.state import EntityState, Vector
from interceptor.core.world import Entity

__all__ = ["Track", "TrackSource", "TruthTrack"]

_EPS = 1e-12


@dataclass(frozen=True)
class Track:
    """Estimated relative geometry between the missile and its target.

    Attributes:
        time: When this estimate applies, seconds.
        relative_position: ``p_target - p_missile`` in the world frame, metres.
        relative_velocity: ``v_target - v_missile`` in the world frame, m/s.
        target_acceleration: Estimated target acceleration, m/s^2. ``None``
            until an estimator that can produce it exists (Phase 5); augmented
            proportional navigation needs it.
        valid: ``False`` when there is no usable measurement — outside the
            seeker's field of view, below its detection threshold, or otherwise
            dropped. A dropout is an ordinary value flowing through the system,
            not an exception.
    """

    time: float
    relative_position: Vector
    relative_velocity: Vector
    target_acceleration: Vector | None = None
    valid: bool = True

    @property
    def range(self) -> float:
        """Distance to the target, metres."""
        return float(np.linalg.norm(self.relative_position))

    @property
    def line_of_sight(self) -> Vector:
        """Unit vector from the missile towards the target."""
        distance = self.range
        if distance < _EPS:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.relative_position / distance, dtype=np.float64)

    @property
    def closing_speed(self) -> float:
        """Rate at which range is shrinking, m/s. Positive while closing.

        This is ``V_c`` in the guidance literature: the negative of the range
        rate, so that a healthy intercept has a positive value.
        """
        distance = self.range
        if distance < _EPS:
            return 0.0
        return -float(np.dot(self.relative_position, self.relative_velocity)) / distance

    @property
    def los_rate_vector(self) -> Vector:
        """The line-of-sight rotation vector, rad/s.

        ``Omega = (r x v) / (r . r)``. Its magnitude is the line-of-sight rate
        that proportional navigation drives to zero, and its direction is the
        axis the sightline is rotating about. This is the single most important
        quantity in the whole project — everything the sensing chain does exists
        to estimate it well.
        """
        r = self.relative_position
        denominator = float(np.dot(r, r))
        if denominator < _EPS:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(np.cross(r, self.relative_velocity) / denominator, dtype=np.float64)

    @property
    def los_rate(self) -> float:
        """Magnitude of the line-of-sight rate, rad/s."""
        return float(np.linalg.norm(self.los_rate_vector))

    @property
    def time_to_go(self) -> float:
        """Crude time until closest approach, seconds.

        ``range / closing_speed``. Exact only for a constant closing velocity,
        which is close enough for a terminal engagement and is what the
        classical guidance derivations assume. Infinite when not closing.
        """
        closing = self.closing_speed
        if closing <= _EPS:
            return float("inf")
        return self.range / closing


class TrackSource(ABC):
    """Produces a :class:`Track` for the guidance law on demand.

    One implementation now (truth), and from Phase 4 a second one built from a
    seeker and a filter. The guidance law cannot tell them apart, which is what
    makes "run this law on perfect information" a one-line debugging move for
    the rest of the project.
    """

    @abstractmethod
    def update(self, t: float, missile: EntityState) -> Track:
        """Return the best current estimate of the engagement geometry."""


class TruthTrack(TrackSource):
    """A perfect track, read straight out of the world.

    No noise, no latency, no field of view, no dropouts. Useful for exactly two
    things: getting a guidance law working before the seeker exists, and
    afterwards as the baseline every degraded configuration is measured against.

    It holds the target *entity*, not a copy of its state. An entity's state is
    replaced by the integrator on every step, so keeping the entity means the
    track is always current; keeping a state would silently freeze the target at
    its starting position — a bug that produces a missile confidently
    intercepting empty air.
    """

    def __init__(self, target: Entity) -> None:
        self.target = target

    def update(self, t: float, missile: EntityState) -> Track:
        target = self.target.state
        return Track(
            time=t,
            relative_position=target.pos - missile.pos,
            relative_velocity=target.vel - missile.vel,
            valid=True,
        )
