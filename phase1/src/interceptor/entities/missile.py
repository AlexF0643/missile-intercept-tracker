"""The missile: a point mass under thrust, drag, gravity and guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.propulsion import Motor
from interceptor.core.frames import unit
from interceptor.core.state import EntityState, Vector
from interceptor.core.world import Entity, air_density

if TYPE_CHECKING:
    from interceptor.core.world import World

__all__ = ["Missile"]

# Below this speed the velocity vector is too short to point the thrust with.
_MIN_STEERING_SPEED = 1.0


class Missile(Entity):
    """A 3-DOF missile.

    Thrust acts along the velocity vector, which is the zero-angle-of-attack
    assumption stated in :mod:`interceptor.core.frames`. Below a walking pace
    the velocity has no usable direction, so thrust falls back to the launch
    direction — otherwise a missile starting from rest would have nothing to
    point at.

    ``guidance_acceleration`` is the lateral command held between guidance
    ticks. Nothing writes to it in Phase 1, so it stays zero and the missile
    flies ballistically; from Phase 2 the guidance law sets it at 100 Hz and the
    integrator sees it as a constant over the ten physics steps in between.
    """

    def __init__(
        self,
        name: str,
        state: EntityState,
        *,
        motor: Motor | None = None,
        aero: Aerodynamics | None = None,
        launch_time: float = 0.0,
        launch_direction: Vector | None = None,
    ) -> None:
        super().__init__(name, state)
        self.motor = motor if motor is not None else Motor()
        self.aero = aero if aero is not None else Aerodynamics()
        self.launch_time = launch_time
        self.guidance_acceleration: Vector = np.zeros(3, dtype=np.float64)

        if launch_direction is not None:
            self._launch_direction = unit(np.asarray(launch_direction, dtype=np.float64))
        elif state.speed >= _MIN_STEERING_SPEED:
            self._launch_direction = unit(state.vel)
        else:
            # Straight up is the only defensible default for a body at rest.
            self._launch_direction = np.array([0.0, 0.0, 1.0])

    def thrust_direction(self, state: EntityState) -> Vector:
        """Unit vector the motor pushes along."""
        if state.speed < _MIN_STEERING_SPEED:
            return self._launch_direction
        return unit(state.vel)

    def acceleration(self, t: float, state: EntityState, world: World) -> Vector:
        acceleration = world.gravity_vector()

        if world.config.enable_drag:
            density = air_density(state.altitude)
            acceleration = acceleration + self.aero.drag_acceleration(
                state.vel, state.mass, density
            )

        thrust = self.motor.thrust(t - self.launch_time)
        if thrust > 0.0:
            acceleration = acceleration + (thrust / state.mass) * self.thrust_direction(state)

        return np.asarray(acceleration + self.guidance_acceleration, dtype=np.float64)

    def mass_flow(self, t: float, state: EntityState) -> float:
        del state
        return self.motor.mass_flow(t - self.launch_time)
