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

__all__ = [
    "Manoeuvre",
    "Target",
    "barrel_roll",
    "break_turn",
    "jink",
    "straight_and_level",
    "weave",
]

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


def _lift_axis_of(velocity: Vector) -> Vector:
    """Unit vector "upwards" relative to the flight path.

    Perpendicular to both the direction of travel and the horizontal right, so
    together the three make a right-handed frame riding with the target. This is
    the axis a pull-up acts along, and having it is what lets a manoeuvre be
    flown in a plane other than the horizontal.
    """
    forward = unit(velocity)
    return np.asarray(np.cross(_horizontal_right_of(velocity), forward), dtype=np.float64)


def _manoeuvre_axis(velocity: Vector, bank_deg: float) -> Vector:
    """The direction a banked manoeuvre pulls in.

    ``bank_deg`` is measured from the horizontal, exactly as a pilot would mean
    it: 0 is a flat turn, 90 is a pure pull-up or push-over, 45 is a climbing
    turn. Anything other than 0 or 180 takes the target out of the horizontal
    plane, which is where a plan view stops being able to show what happened.
    """
    angle = np.deg2rad(bank_deg)
    right = _horizontal_right_of(velocity)
    lift = _lift_axis_of(velocity)
    return np.asarray(np.cos(angle) * right + np.sin(angle) * lift, dtype=np.float64)


def weave(amplitude_g: float, period: float, bank_deg: float = 0.0) -> Manoeuvre:
    """A sinusoidal weave — the classic evasive manoeuvre.

    Args:
        amplitude_g: Peak lateral acceleration, in multiples of g.
        period: Seconds for one full left-right cycle.
        bank_deg: The plane to weave in, measured from horizontal. 0 is the
            traditional flat weave; 90 weaves vertically, porpoising up and
            down; anything between does both at once.

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
        return magnitude * _manoeuvre_axis(state.vel, bank_deg)

    return manoeuvre


def break_turn(amplitude_g: float, start_time: float = 0.0, bank_deg: float = 0.0) -> Manoeuvre:
    """A sustained hard turn in one direction, beginning at ``start_time``.

    A late break is the hardest case in the whole project: it arrives when the
    missile is slowest, has least time to respond, and its own available g has
    decayed with its speed.

    ``bank_deg`` banks the break out of the horizontal — at 90 it becomes a pure
    pull-up, at -90 a split-S downwards, which is what a pilot with altitude to
    trade would actually do.
    """
    peak = amplitude_g * STANDARD_GRAVITY

    def manoeuvre(t: float, state: EntityState) -> Vector:
        if t < start_time:
            return np.zeros(3, dtype=np.float64)
        return peak * _manoeuvre_axis(state.vel, bank_deg)

    return manoeuvre


def barrel_roll(amplitude_g: float, period: float) -> Manoeuvre:
    """A constant-g pull whose direction rotates about the flight path.

    The target corkscrews. Acceleration magnitude never changes — only where it
    points — so the path is a helix about the original track rather than a
    weave in any one plane.

    This is the manoeuvre that makes a three-dimensional view worth having. A
    weave and a barrel roll look nearly identical in plan view and are quite
    different things, and a guidance law that copes with one need not cope with
    the other: the line-of-sight rate never settles in any plane, so an
    estimator cannot smooth it away in one axis and be right about the rest.
    """
    if period <= 0.0:
        msg = f"barrel roll period must be positive, got {period}"
        raise ValueError(msg)
    omega = 2.0 * np.pi / period
    peak = amplitude_g * STANDARD_GRAVITY

    def manoeuvre(t: float, state: EntityState) -> Vector:
        angle = omega * t
        direction = np.cos(angle) * _horizontal_right_of(state.vel) + np.sin(angle) * _lift_axis_of(
            state.vel
        )
        return np.asarray(peak * direction, dtype=np.float64)

    return manoeuvre


def jink(amplitude_g: float, interval: float = 1.5, seed: int = 0) -> Manoeuvre:
    """Hard pulls in unpredictable directions, changing every ``interval``.

    What a pilot who knows a missile is coming actually does: not a tidy
    sinusoid but a series of violent, uncorrelated breaks, chosen precisely so
    that no filter can learn the pattern. Where a weave punishes an estimator
    that smooths too hard, a jink punishes any estimator at all — a
    constant-acceleration model is wrong the instant the direction changes, and
    there is nothing in the history to predict the next one from.

    Deterministic given ``seed``: the directions are drawn once, up front, so
    the same seed always flies the same path and a result can be reproduced.
    """
    if interval <= 0.0:
        msg = f"jink interval must be positive, got {interval}"
        raise ValueError(msg)
    peak = amplitude_g * STANDARD_GRAVITY

    # Drawn ahead of time rather than on demand: a manoeuvre is called from
    # inside the integrator, four times per step, at times that are not
    # monotonic. Sampling there would make the path depend on the integration
    # scheme, which would be an excellent way to produce an unrepeatable bug.
    angles = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=4096)

    def manoeuvre(t: float, state: EntityState) -> Vector:
        angle = float(angles[int(t / interval) % angles.size])
        direction = np.cos(angle) * _horizontal_right_of(state.vel) + np.sin(angle) * _lift_axis_of(
            state.vel
        )
        return np.asarray(peak * direction, dtype=np.float64)

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

    def commanded_manoeuvre(self, t: float) -> Vector:
        """A scripted target's acceleration is exactly its manoeuvre."""
        return self.manoeuvre(t, self.state)
