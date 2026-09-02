"""The autopilot: what the missile actually does about what it was told.

Two effects separate the command from the response, and both matter more than
their size suggests.

**The limiter.** A guidance law will happily ask for 200 g in the last tenth of
a second. The airframe cannot deliver it, and pretending otherwise turns every
guidance law into a perfect one.

**The lag.** Fins take time to move and the airframe takes time to develop the
angle of attack that produces the force. A first-order lag with a time constant
of about 0.2 s captures that, and it is responsible for a large share of real
miss distance: a command issued 0.2 s before impact has barely begun to take
effect by the time it matters.

Keeping this separate from the guidance law means the plots can show demanded
and achieved acceleration side by side, and saturation becomes visible rather
than silent.
"""

from __future__ import annotations

import numpy as np

from interceptor.core.state import Vector

__all__ = ["Autopilot", "clamp_magnitude"]


def clamp_magnitude(vector: Vector, limit: float) -> Vector:
    """Scale ``vector`` down to ``limit`` if it is longer, preserving direction."""
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= limit or magnitude < 1e-12:
        return np.asarray(vector, dtype=np.float64)
    return np.asarray(vector * (limit / magnitude), dtype=np.float64)


class Autopilot:
    """First-order lag plus an acceleration limiter.

    Args:
        time_constant: Seconds for the achieved acceleration to reach about 63%
            of a step command. Zero makes the response instantaneous, which is
            useful for isolating guidance behaviour in a test.
    """

    def __init__(self, time_constant: float = 0.20) -> None:
        if time_constant < 0.0:
            msg = f"time_constant cannot be negative, got {time_constant}"
            raise ValueError(msg)
        self.time_constant = time_constant
        self.achieved: Vector = np.zeros(3, dtype=np.float64)
        self.demanded: Vector = np.zeros(3, dtype=np.float64)

    def update(self, commanded: Vector, dt: float, limit: float) -> Vector:
        """Advance one guidance tick and return the acceleration to apply.

        Args:
            commanded: What the guidance law asked for, m/s^2.
            dt: Time since the last guidance tick, seconds.
            limit: The most the airframe can currently produce, m/s^2.
        """
        self.demanded = clamp_magnitude(commanded, limit)

        if self.time_constant <= 0.0:
            self.achieved = self.demanded
        else:
            # Explicit first-order lag. The fraction is capped at 1 so a
            # guidance interval longer than the time constant cannot overshoot.
            fraction = min(dt / self.time_constant, 1.0)
            self.achieved = self.achieved + fraction * (self.demanded - self.achieved)

        return self.achieved

    @property
    def saturated(self) -> bool:
        """True when the limiter is currently cutting the command down."""
        return bool(np.linalg.norm(self.demanded) > 0.0) and not np.allclose(
            self.demanded, self.achieved, rtol=0.0, atol=1e-9
        )
