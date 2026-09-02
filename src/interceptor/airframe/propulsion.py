"""Rocket motor: a thrust schedule and the mass it costs to produce it."""

from __future__ import annotations

from dataclasses import dataclass

from interceptor.core.world import STANDARD_GRAVITY

__all__ = ["Motor"]


@dataclass(frozen=True)
class Motor:
    """A two-phase solid motor: a short high boost, then an optional sustain.

    Defaults describe an inert body — no thrust, no mass loss — which is what
    the ballistic tests need.

    Attributes:
        boost_thrust: Thrust during the boost phase, newtons.
        boost_duration: Length of the boost phase, seconds.
        sustain_thrust: Thrust after boost, newtons.
        sustain_duration: Length of the sustain phase, seconds.
        specific_impulse: Isp in seconds. Converts thrust into propellant flow
            via ``mdot = F / (Isp * g0)``. Around 240 s is typical of a solid
            propellant.
    """

    boost_thrust: float = 0.0
    boost_duration: float = 0.0
    sustain_thrust: float = 0.0
    sustain_duration: float = 0.0
    specific_impulse: float = 240.0

    def __post_init__(self) -> None:
        if self.specific_impulse <= 0.0:
            msg = f"specific_impulse must be positive, got {self.specific_impulse}"
            raise ValueError(msg)
        if self.boost_duration < 0.0 or self.sustain_duration < 0.0:
            msg = "motor phase durations cannot be negative"
            raise ValueError(msg)

    @property
    def burn_time(self) -> float:
        """Total time producing thrust, seconds."""
        return self.boost_duration + self.sustain_duration

    def thrust(self, time_since_launch: float) -> float:
        """Thrust at a time measured from launch, newtons.

        Zero before launch and after burnout. The transition between phases is a
        step, which is the honest shape for a solid motor and costs the
        integrator nothing at a 1 ms step.
        """
        if time_since_launch < 0.0:
            return 0.0
        if time_since_launch < self.boost_duration:
            return self.boost_thrust
        if time_since_launch < self.burn_time:
            return self.sustain_thrust
        return 0.0

    def mass_flow(self, time_since_launch: float) -> float:
        """Rate of mass change, kg/s. Negative while burning, zero otherwise."""
        return -self.thrust(time_since_launch) / (self.specific_impulse * STANDARD_GRAVITY)

    @property
    def total_impulse(self) -> float:
        """Impulse delivered over the whole burn, newton-seconds."""
        return self.boost_thrust * self.boost_duration + self.sustain_thrust * self.sustain_duration

    @property
    def propellant_mass(self) -> float:
        """Propellant consumed over the whole burn, kg."""
        return self.total_impulse / (self.specific_impulse * STANDARD_GRAVITY)
