"""Finding the point of closest approach, exactly.

This is the single most important piece of measurement code in the project, and
the easiest to get quietly wrong.

At a 1 ms step and a closing speed near 1 km/s, the missile moves a full metre
between samples. Reporting the smallest sampled separation therefore quantises
miss distance to roughly a metre — which is the same order as the difference
between a good guidance law and a mediocre one. Every comparison in Phase 3
onward would be measuring the sampling grid rather than the guidance.

The fix is to stop treating the samples as the answer. Detect the moment the
range stops shrinking, then solve for the true minimum *between* the two
straddling steps. Over one millisecond the relative velocity is effectively
constant, so the separation is a parabola in time and its minimum is exact:

    t* = -(r . v) / (v . v)

evaluated at the step where closing ceased, giving a small negative offset back
to the true closest approach. The miss distance is then ``|r + v t*|``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interceptor.core.world import World

__all__ = ["ClosestApproach", "Intercept"]

_EPS = 1e-12


@dataclass(frozen=True)
class Intercept:
    """The outcome of an engagement."""

    time: float
    """Simulated time of closest approach, seconds."""

    miss_distance: float
    """Separation at closest approach, metres."""

    closing_speed: float
    """Relative speed at closest approach, m/s."""

    hit: bool
    """Whether the miss distance fell inside the lethal radius."""

    def __str__(self) -> str:
        verdict = "HIT" if self.hit else "miss"
        return (
            f"{verdict}: {self.miss_distance:.3f} m at t = {self.time:.3f} s, "
            f"closing at {self.closing_speed:.0f} m/s"
        )


class ClosestApproach:
    """Stop condition that ends the run at closest approach and measures it.

    Usable directly as the ``stop`` argument to
    :func:`~interceptor.sim.engagement.run`. After the run, :attr:`result` holds
    the :class:`Intercept`, or ``None`` if the missile never stopped closing.

    Args:
        missile: Name of the pursuing entity.
        target: Name of the pursued entity.
        lethal_radius: Miss distances at or below this count as a hit. A few
            metres is representative of a proximity-fused warhead; zero demands
            a direct impact.
    """

    def __init__(self, missile: str, target: str, lethal_radius: float = 5.0) -> None:
        self.missile = missile
        self.target = target
        self.lethal_radius = lethal_radius
        self.result: Intercept | None = None
        self._was_closing = False

    def __call__(self, t: float, world: World) -> str | None:
        r = world[self.target].state.pos - world[self.missile].state.pos
        v = world[self.target].state.vel - world[self.missile].state.vel

        range_rate = float(np.dot(r, v))  # negative while closing
        closing = range_rate < 0.0

        if self._was_closing and not closing:
            self.result = self._solve(t, r, v)
            return "closest approach"

        self._was_closing = closing
        return None

    def _solve(self, t: float, r: np.ndarray, v: np.ndarray) -> Intercept:
        """Interpolate back to the exact minimum of |r(t)|."""
        speed_squared = float(np.dot(v, v))
        offset = 0.0 if speed_squared < _EPS else -float(np.dot(r, v)) / speed_squared

        closest = r + offset * v
        distance = float(np.linalg.norm(closest))
        return Intercept(
            time=t + offset,
            miss_distance=distance,
            closing_speed=float(np.linalg.norm(v)),
            hit=distance <= self.lethal_radius,
        )
