"""The world: truth, and the fixed-step loop that advances it.

Entities declare the forces acting on them; the world integrates. That split is
what keeps a missile's ``acceleration`` method testable in isolation — it is a
pure function of time and state, with no stepping logic inside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from interceptor.core.integrate import rk4
from interceptor.core.state import EntityState, Vector

__all__ = [
    "SCALE_HEIGHT",
    "SEA_LEVEL_DENSITY",
    "STANDARD_GRAVITY",
    "Entity",
    "World",
    "WorldConfig",
    "air_density",
]

#: Standard gravitational acceleration, m/s^2.
STANDARD_GRAVITY: Final = 9.80665

#: ISA sea-level air density, kg/m^3.
SEA_LEVEL_DENSITY: Final = 1.225

#: Exponential atmosphere scale height, metres.
SCALE_HEIGHT: Final = 8500.0


def air_density(altitude: float) -> float:
    """Air density at an altitude, by the exponential atmosphere approximation.

    ``rho = 1.225 * exp(-h / 8500)``. Accurate enough below about 20 km for a
    drag model that already assumes a single drag coefficient, and it has the
    virtue of being one line with no lookup table. Below sea level the density
    is held at its sea-level value rather than growing without bound.
    """
    return SEA_LEVEL_DENSITY * float(np.exp(-max(altitude, 0.0) / SCALE_HEIGHT))


@dataclass(frozen=True)
class WorldConfig:
    """Which physics the world applies.

    The switches exist for testing. Turning gravity and drag off leaves a body
    travelling in a straight line at constant speed, and turning drag off alone
    leaves a trajectory with a closed-form solution to compare against — which
    is exactly what the Phase 1 exit criterion does.
    """

    gravity: float = STANDARD_GRAVITY
    enable_gravity: bool = True
    enable_drag: bool = True


class Entity(ABC):
    """A body occupying the world.

    Subclasses declare what accelerates them and, if they burn propellant, how
    fast they lose mass. They never integrate anything themselves.
    """

    def __init__(self, name: str, state: EntityState) -> None:
        self.name = name
        self.state = state

    @abstractmethod
    def acceleration(self, t: float, state: EntityState, world: World) -> Vector:
        """Total acceleration in the world frame, m/s^2."""

    def mass_flow(self, t: float, state: EntityState) -> float:
        """Rate of change of mass, kg/s. Negative while burning; zero by default."""
        del t, state
        return 0.0

    def commanded_manoeuvre(self, t: float) -> Vector | None:
        """The acceleration this body is commanding of itself, if it knows.

        ``None`` by default, because most entities do not have a plan — a
        missile's acceleration is the outcome of thrust, drag and guidance
        rather than something it decides. A scripted target does have one, and
        overrides this.

        It exists so that a *truth* track can report the target's real
        acceleration to augmented proportional navigation. That separates two
        questions which would otherwise be tangled: whether the augmented term
        is a good idea, and whether the filter's estimate of target acceleration
        is good enough to feed it. Answering them apart is the whole reason the
        project keeps a perfect-information baseline.
        """
        del t
        return None

    def update_guidance(self, t: float, dt: float) -> None:
        """Hook called at the guidance rate, not the physics rate.

        Whatever this sets is then held constant while the integrator takes the
        steps in between — a zero-order hold, which is what a real digital
        autopilot does. A no-op for anything that is not guided.
        """
        del t, dt

    def commanded_acceleration(self) -> float:
        """Magnitude of the acceleration guidance asked for, m/s^2."""
        return 0.0

    def achieved_acceleration(self) -> float:
        """Magnitude of the acceleration actually applied, m/s^2.

        Differs from :meth:`commanded_acceleration` when the limiter is
        saturating or the autopilot lag has not caught up. Recording both is
        what makes those two failure modes visible in a plot.
        """
        return 0.0

    def acceleration_limit(self) -> float:
        """The most lateral acceleration this body could produce now, m/s^2."""
        return 0.0


class World:
    """Holds the entities and advances them all by one fixed step at a time."""

    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config if config is not None else WorldConfig()
        self._entities: dict[str, Entity] = {}

    def add(self, entity: Entity) -> Entity:
        """Register an entity. Returns it, so it can be added and bound in one line."""
        if entity.name in self._entities:
            msg = f"an entity named {entity.name!r} is already in this world"
            raise ValueError(msg)
        self._entities[entity.name] = entity
        return entity

    def __getitem__(self, name: str) -> Entity:
        return self._entities[name]

    @property
    def names(self) -> tuple[str, ...]:
        """Entity names, in insertion order."""
        return tuple(self._entities)

    @property
    def entities(self) -> tuple[Entity, ...]:
        """The entities, in insertion order."""
        return tuple(self._entities.values())

    def gravity_vector(self) -> Vector:
        """Gravitational acceleration in the world frame — straight down, or zero."""
        if not self.config.enable_gravity:
            return np.zeros(3, dtype=np.float64)
        return np.array([0.0, 0.0, -self.config.gravity], dtype=np.float64)

    def _derivative_of(self, entity: Entity) -> Callable[[float, Vector], Vector]:
        """Build the ``f(t, y)`` the integrator needs for one entity."""

        def derivative(t: float, y: Vector) -> Vector:
            state = EntityState.from_array(y)
            return np.concatenate(
                (
                    state.vel,
                    entity.acceleration(t, state, self),
                    np.array([entity.mass_flow(t, state)], dtype=np.float64),
                )
            )

        return derivative

    def step(self, t: float, dt: float) -> None:
        """Advance every entity by ``dt`` using RK4.

        All entities are stepped from the same start time, so nothing sees a
        half-updated world mid-step.
        """
        for entity in self._entities.values():
            advanced = rk4(self._derivative_of(entity), t, entity.state.to_array(), dt)
            entity.state = EntityState.from_array(advanced)
