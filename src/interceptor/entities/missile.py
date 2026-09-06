"""The missile: a point mass under thrust, drag, gravity and guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.autopilot import Autopilot, clamp_magnitude
from interceptor.airframe.propulsion import Motor
from interceptor.core.frames import unit
from interceptor.core.state import EntityState, Vector, magnitude
from interceptor.core.world import Entity, air_density
from interceptor.guidance.base import GuidanceLaw
from interceptor.sensing.track import Track, TrackSource

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
    ticks. With no guidance law attached it stays zero and the missile flies
    ballistically; otherwise the law sets it at the guidance rate and the
    integrator sees it as a constant over the ten physics steps in between.

    Guidance is supplied as two independent pieces — a
    :class:`~interceptor.sensing.track.TrackSource` that says what the missile
    believes, and a :class:`~interceptor.guidance.base.GuidanceLaw` that decides
    what to do about it. Swapping either one leaves the other untouched, which
    is how the same law can be run against perfect information and against a
    noisy seeker without changing a line of it.
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
        guidance: GuidanceLaw | None = None,
        track_source: TrackSource | None = None,
        autopilot: Autopilot | None = None,
    ) -> None:
        super().__init__(name, state)
        self.motor = motor if motor is not None else Motor()
        self.aero = aero if aero is not None else Aerodynamics()
        self.launch_time = launch_time
        self.guidance = guidance
        self.track_source = track_source
        self.autopilot = autopilot if autopilot is not None else Autopilot()

        self.guidance_acceleration: Vector = np.zeros(3, dtype=np.float64)
        self.commanded: Vector = np.zeros(3, dtype=np.float64)
        self.latest_track: Track | None = None

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

        # The guidance command is needed before drag, not after: turning costs
        # drag, so how hard the missile is pulling right now changes how fast it
        # is slowing down. Computing drag first and steering afterwards would
        # let the airframe manoeuvre for free.
        lateral = self.applied_guidance(state)

        if world.config.enable_drag:
            density = air_density(state.altitude)
            acceleration = acceleration + self.aero.drag_acceleration(
                state.vel,
                state.mass,
                density,
                lateral_acceleration=magnitude(lateral),
            )

        thrust = self.motor.thrust(t - self.launch_time)
        if thrust > 0.0:
            acceleration = acceleration + (thrust / state.mass) * self.thrust_direction(state)

        return np.asarray(acceleration + lateral, dtype=np.float64)

    def applied_guidance(self, state: EntityState) -> Vector:
        """The held guidance command, capped by what the air can supply *now*.

        The autopilot already clamped the command at the last guidance tick, but
        that limit was computed from the state at that moment. Ten physics steps
        later the missile is slower and lower, and its real limit has fallen —
        so a command that was achievable when issued may not be by the time it
        is applied. Re-capping here against the current state means the missile
        can never pull more than the air allows, which is the physical truth and
        also what makes the recorded trace trustworthy.
        """
        limit = self.aero.available_lateral_acceleration(
            state.speed, state.mass, air_density(state.altitude)
        )
        return clamp_magnitude(self.guidance_acceleration, limit)

    def mass_flow(self, t: float, state: EntityState) -> float:
        del state
        return self.motor.mass_flow(t - self.launch_time)

    def available_lateral_acceleration(self) -> float:
        """What the airframe can currently produce, m/s^2."""
        return self.aero.available_lateral_acceleration(
            self.state.speed, self.state.mass, air_density(self.state.altitude)
        )

    def update_guidance(self, t: float, dt: float) -> None:
        """Run one guidance cycle: read the track, command, limit, hold.

        The order matters. The law is given the track and produces an unclamped
        command; the autopilot then applies the airframe's limit and its lag.
        Recording the command before the clamp is what makes saturation legible
        afterwards — a law that spends the endgame asking for three times what
        the airframe can give has failed in a specific, diagnosable way.
        """
        if self.guidance is None or self.track_source is None:
            return

        track = self.track_source.update(t, self.state)
        self.latest_track = track
        self.commanded = self.guidance.command(track, self.state)
        self.guidance_acceleration = self.autopilot.update(
            self.commanded, dt, self.available_lateral_acceleration()
        )

    def commanded_acceleration(self) -> float:
        return float(np.linalg.norm(self.commanded))

    def achieved_acceleration(self) -> float:
        return float(np.linalg.norm(self.applied_guidance(self.state)))

    def acceleration_limit(self) -> float:
        return self.available_lateral_acceleration()
