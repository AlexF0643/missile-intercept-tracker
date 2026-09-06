"""Point-mass aerodynamics: drag, and the limit on how hard the body can turn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interceptor.core.state import Vector, magnitude
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
        peak_lift_angle_deg: The angle of attack at which that peak lift is
            reached. Only used to work out what turning costs — see
            :attr:`induced_drag_factor`.
    """

    drag_coefficient: float = 0.30
    reference_area: float = 0.02
    max_lateral_g: float = 30.0
    max_lift_coefficient: float = 2.50
    peak_lift_angle_deg: float = 25.0

    @property
    def induced_drag_factor(self) -> float:
        """``k`` in ``Cd = Cd0 + k * Cn^2`` — the price of turning.

        Not a free constant: it falls out of two numbers the airframe has
        already declared. A body at incidence produces its normal force
        perpendicular to its own axis rather than to the flight path, so a
        component of that force points backwards. To first order the streamwise
        share is ``Cn * sin(alpha) ~ Cn * alpha``, and with ``Cn ~ Cn_alpha *
        alpha`` that is ``Cn^2 / Cn_alpha``. Hence ``k = 1 / Cn_alpha``, and the
        lift-curve slope is just the peak lift divided by the angle it is
        reached at.

        With the defaults — 2.5 at 25 degrees — this gives ``k ~ 0.175``, so an
        airframe pulling its full lift coefficient sees an induced ``Cd`` near
        1.1 against a zero-lift 0.30. Manoeuvring hard roughly quadruples the
        drag, which is the single most important thing this correction adds:
        before it, the missile turned for free.

        **This is an engineering estimate, not a citation.** The form is
        standard and the magnitude is right, but a real design would take
        ``Cn_alpha`` and the induced-drag efficiency from a wind-tunnel database
        — Fleeman's *Tactical Missile Design* or USAF DATCOM. Treat the number
        as a placeholder with the correct physics behind it, and replace it if
        this project ever needs to claim absolute accuracy rather than
        comparative honesty.
        """
        return float(np.deg2rad(self.peak_lift_angle_deg)) / self.max_lift_coefficient

    def dynamic_pressure(self, speed: float, density: float) -> float:
        """``q = 0.5 * rho * V^2``, in pascals."""
        return 0.5 * density * speed * speed

    def normal_force_coefficient(
        self, lateral_acceleration: float, speed: float, mass: float, density: float
    ) -> float:
        """``Cn`` needed to produce a given lateral acceleration."""
        pressure = self.dynamic_pressure(speed, density) * self.reference_area
        if pressure <= 0.0:
            return 0.0
        return mass * abs(lateral_acceleration) / pressure

    def total_drag_coefficient(
        self, lateral_acceleration: float, speed: float, mass: float, density: float
    ) -> float:
        """Zero-lift drag plus the induced drag of the turn being flown."""
        coefficient = self.normal_force_coefficient(lateral_acceleration, speed, mass, density)
        return self.drag_coefficient + self.induced_drag_factor * coefficient * coefficient

    def drag_acceleration(
        self,
        velocity: Vector,
        mass: float,
        density: float,
        lateral_acceleration: float = 0.0,
    ) -> Vector:
        """Deceleration from drag, opposing the velocity, m/s^2.

        ``a = -(0.5 * rho * Cd * S / m) * |v| * v``. Written against the vector
        rather than the speed so that no direction has to be reconstructed, and
        so a zero velocity gives exactly zero drag with no special case.

        ``lateral_acceleration`` is what the airframe is currently pulling, and
        it adds the induced term. Defaulting it to zero keeps an unguided body
        exactly as it was: a ballistic trajectory is unchanged, which is why the
        Phase 1 parabola test still holds to 1e-10.
        """
        speed = magnitude(velocity)
        coefficient = self.total_drag_coefficient(lateral_acceleration, speed, mass, density)
        factor = 0.5 * density * coefficient * self.reference_area * speed / mass
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
