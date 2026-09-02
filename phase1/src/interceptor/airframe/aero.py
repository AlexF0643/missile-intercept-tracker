"""Point-mass aerodynamics: drag, and the limit on how hard the body can turn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interceptor.core.state import Vector
from interceptor.core.world import STANDARD_GRAVITY

__all__ = ["Aerodynamics"]


@dataclass(frozen=True)
class Aerodynamics:
    """A drag coefficient, a reference area, and a manoeuvre envelope.

    Attributes:
        drag_coefficient: Cd, dimensionless. Held constant — a real missile's Cd
            varies sharply through the transonic region, and modelling that
            would need a lookup table this project does not have.
        reference_area: S, m^2. The cross-section drag is computed against.
        max_lateral_g: The structural limit, in multiples of g.
        max_lift_coefficient: Peak C_L the airframe can generate at high angle
            of attack, referenced to ``reference_area``. Around 2.5 is typical
            of a finned missile — well above an aircraft wing's ~1.5, because
            the body and tail surfaces contribute and the airframe is flown to
            angles of attack an aircraft would never see.
    """

    drag_coefficient: float = 0.30
    reference_area: float = 0.02
    max_lateral_g: float = 30.0
    max_lift_coefficient: float = 2.50

    def dynamic_pressure(self, speed: float, density: float) -> float:
        """``q = 0.5 * rho * V^2``, in pascals."""
        return 0.5 * density * speed * speed

    def drag_acceleration(self, velocity: Vector, mass: float, density: float) -> Vector:
        """Deceleration from drag, opposing the velocity, m/s^2.

        ``a = -(0.5 * rho * Cd * S / m) * |v| * v``. Written against the vector
        rather than the speed so that no direction has to be reconstructed, and
        so a zero velocity gives exactly zero drag with no special case.
        """
        speed = float(np.linalg.norm(velocity))
        factor = 0.5 * density * self.drag_coefficient * self.reference_area * speed / mass
        return np.asarray(-factor * velocity, dtype=np.float64)

    def available_lateral_acceleration(self, speed: float, mass: float, density: float) -> float:
        """The most lateral acceleration this body can actually produce, m/s^2.

        The lesser of the structural limit and what the air can supply. The
        second term is why a missile that has bled its speed cannot pull its
        rated g — it collapses as ``V^2``, so a slow missile at altitude is a
        poor turner regardless of what its airframe is rated for. Fast and low,
        the structural limit binds; slow or high, the air does. Unused in
        Phase 1; it becomes the autopilot's clamp in Phase 3.
        """
        structural = self.max_lateral_g * STANDARD_GRAVITY
        aerodynamic = (
            self.max_lift_coefficient
            * self.dynamic_pressure(speed, density)
            * self.reference_area
            / mass
        )
        return min(structural, aerodynamic)
