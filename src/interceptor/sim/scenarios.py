"""Standard engagement geometries.

Three cases, chosen because they stress a guidance law in different ways.

**Head-on** is the easy one: the target flies straight at the missile, the
line of sight barely rotates, and almost anything intercepts.

**Crossing** is the discriminating case. The target flies at right angles to
the sightline, so the bearing sweeps steadily and a law that steers at the
target's present position is always aiming behind it. This is where pure
pursuit and proportional navigation part company.

**Tail chase** is the hard one: the missile has to run the target down from
behind, closing speed is low, and there is a great deal of time for small
errors to accumulate.

Defining them here rather than inside each test means a change to the missile's
motor shows up consistently everywhere, and a comparison between two guidance
laws is genuinely like-for-like.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.autopilot import Autopilot
from interceptor.airframe.propulsion import Motor
from interceptor.core.state import EntityState, Vector
from interceptor.core.world import World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.entities.target import Manoeuvre, Target, straight_and_level
from interceptor.guidance.base import GuidanceLaw
from interceptor.sensing.track import TruthTrack
from interceptor.sim.intercept import ClosestApproach

__all__ = ["Scenario", "crossing", "head_on", "tail_chase"]

MISSILE = "missile"
TARGET = "target"

#: A boost-only solid motor: 20 kN for 2.5 s, taking an 85 kg round from launch
#: speed to roughly Mach 2 before it begins to coast.
DEFAULT_MOTOR = Motor(boost_thrust=20_000.0, boost_duration=2.5, specific_impulse=240.0)


@dataclass(frozen=True)
class Scenario:
    """A complete engagement, ready to be built and run."""

    name: str
    missile_position: Vector
    missile_velocity: Vector
    target_position: Vector
    target_velocity: Vector
    manoeuvre: Manoeuvre = field(default_factory=straight_and_level)
    missile_mass: float = 85.0
    motor: Motor = DEFAULT_MOTOR
    aero: Aerodynamics = field(default_factory=Aerodynamics)
    autopilot_lag: float = 0.20
    lethal_radius: float = 5.0
    duration: float = 40.0

    def with_manoeuvre(self, manoeuvre: Manoeuvre) -> Scenario:
        """A copy of this scenario with a different target manoeuvre."""
        return replace(self, manoeuvre=manoeuvre)

    def build(self, law: GuidanceLaw | None = None) -> tuple[World, ClosestApproach]:
        """Construct the world and the closest-approach detector.

        Passing ``law=None`` builds an unguided missile, which is how the tests
        confirm that an intercept is caused by the guidance rather than by a
        fortunate initial aim.
        """
        world = World(WorldConfig())

        target = Target(
            TARGET,
            EntityState(pos=self.target_position, vel=self.target_velocity),
            self.manoeuvre,
        )
        world.add(target)

        world.add(
            Missile(
                MISSILE,
                EntityState(
                    pos=self.missile_position,
                    vel=self.missile_velocity,
                    mass=self.missile_mass,
                ),
                motor=self.motor,
                aero=self.aero,
                guidance=law,
                # The track source holds the target entity, so it reports the
                # target's live position rather than a stale copy.
                track_source=TruthTrack(target) if law is not None else None,
                autopilot=Autopilot(self.autopilot_lag),
            )
        )

        return world, ClosestApproach(MISSILE, TARGET, self.lethal_radius)


def _launch_towards(origin: Vector, aim: Vector, speed: float = 60.0) -> Vector:
    """Initial velocity pointing from ``origin`` at ``aim``, off the rail."""
    direction = aim - origin
    return np.asarray(speed * direction / np.linalg.norm(direction), dtype=np.float64)


def head_on(altitude: float = 1000.0, target_speed: float = 250.0) -> Scenario:
    """Target closing straight down the sightline from 8 km."""
    missile_position = np.array([0.0, 0.0, altitude])
    target_position = np.array([0.0, 8000.0, altitude])
    return Scenario(
        name="head-on",
        missile_position=missile_position,
        missile_velocity=_launch_towards(missile_position, target_position),
        target_position=target_position,
        target_velocity=np.array([0.0, -target_speed, 0.0]),
        duration=20.0,
    )


def crossing(altitude: float = 1000.0, target_speed: float = 250.0) -> Scenario:
    """Target crossing the sightline at right angles from 6 km.

    The case that separates the guidance laws. The bearing to the target sweeps
    continuously, so steering at its present position is always steering behind
    it.
    """
    missile_position = np.array([0.0, 0.0, altitude])
    target_position = np.array([0.0, 6000.0, altitude])
    return Scenario(
        name="crossing",
        missile_position=missile_position,
        missile_velocity=_launch_towards(missile_position, target_position),
        target_position=target_position,
        target_velocity=np.array([target_speed, 0.0, 0.0]),
        duration=30.0,
    )


def tail_chase(altitude: float = 1000.0, target_speed: float = 250.0) -> Scenario:
    """Target running away from 3 km — low closing speed, long flight."""
    missile_position = np.array([0.0, 0.0, altitude])
    target_position = np.array([0.0, 3000.0, altitude])
    return Scenario(
        name="tail-chase",
        missile_position=missile_position,
        missile_velocity=_launch_towards(missile_position, target_position),
        target_position=target_position,
        target_velocity=np.array([0.0, target_speed, 0.0]),
        duration=40.0,
    )
