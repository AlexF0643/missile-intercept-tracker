"""The target: a kinematic body following a scripted manoeuvre.

The target is deliberately *not* modelled with gravity and drag. An aircraft
holding altitude is trimmed — its lift already cancels its weight and its thrust
already cancels its drag — so adding those forces to a point mass would produce
a target that falls out of the sky. Treating the manoeuvre command as the total
acceleration is the standard approach in engagement simulation, and it makes the
target's path exactly what the scenario says it is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from interceptor.core.frames import WORLD_UP, unit
from interceptor.core.state import EntityState, Vector
from interceptor.core.world import STANDARD_GRAVITY, Entity

if TYPE_CHECKING:
    from interceptor.core.world import World

__all__ = ["Manoeuvre", "Target", "break_turn", "straight_and_level", "weave"]

#: A manoeuvre maps time and state to a commanded acceleration in the world frame.
Manoeuvre = Callable[[float, EntityState], Vector]


def straight_and_level() -> Manoeuvre:
    """No acceleration at all. The baseline every guidance law must beat."""

    def manoeuvre(t: float, state: EntityState) -> Vector:
        del t, state
        return np.zeros(3, dtype=np.float64)

    return manoeuvre


def _horizontal_right_of(velocity: Vector) -> Vector:
    """Unit vector horizontally to the right of the direction of travel."""
    forward = unit(velocity)
    right = np.cross(forward, WORLD_UP)
    norm = float(np.linalg.norm(right))
    if norm < 1e-9:  # travelling vertically; any horizontal direction will do
        return np.array([1.0, 0.0, 0.0])
    return np.asarray(right / norm, dtype=np.float64)


def weave(amplitude_g: float, period: float) -> Manoeuvre:
    """A sinusoidal horizontal weave — the classic evasive manoeuvre.

    Args:
        amplitude_g: Peak lateral acceleration, in multiples of g.
        period: Seconds for one full left-right cycle.

    The acceleration is phased as a cosine, not a sine, and that is not
    cosmetic. Heading is the integral of lateral acceleration, so a sine
    command integrates to ``1 - cos``, which never changes sign: the target
    would bend to one side and straighten, over and over, without ever crossing
    back. A cosine command integrates to ``sin``, giving a heading that
    oscillates symmetrically about the original track — an actual weave.

    A weave is hard on a guidance law for a specific reason: it keeps the
    line-of-sight rate oscillating, so a filter tuned to smooth measurement
    noise also smooths away the signal it needs.
    """
    if period <= 0.0:
        msg = f"weave period must be positive, got {period}"
        raise ValueError(msg)
    omega = 2.0 * np.pi / period
    peak = amplitude_g * STANDARD_GRAVITY

    def manoeuvre(t: float, state: EntityState) -> Vector:
        magnitude = peak * float(np.cos(omega * t))
        return magnitude * _horizontal_right_of(state.vel)

    return manoeuvre


def break_turn(amplitude_g: float, start_time: float = 0.0) -> Manoeuvre:
    """A sustained hard turn in one direction, beginning at ``start_time``.

    A late break is the hardest case in the whole project: it arrives when the
    missile is slowest, has least time to respond, and its own available g has
    decayed with its speed.
    """
    peak = amplitude_g * STANDARD_GRAVITY

    def manoeuvre(t: float, state: EntityState) -> Vector:
        if t < start_time:
            return np.zeros(3, dtype=np.float64)
        return peak * _horizontal_right_of(state.vel)

    return manoeuvre


class Target(Entity):
    """A body whose acceleration is exactly its commanded manoeuvre."""

    def __init__(
        self,
        name: str,
        state: EntityState,
        manoeuvre: Manoeuvre | None = None,
    ) -> None:
        super().__init__(name, state)
        self.manoeuvre: Manoeuvre = manoeuvre if manoeuvre is not None else straight_and_level()

    def acceleration(self, t: float, state: EntityState, world: World) -> Vector:
        del world  # a kinematic target ignores gravity and drag by design
        return self.manoeuvre(t, state)
