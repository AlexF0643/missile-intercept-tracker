"""Fly one unguided missile and print what happened.

    python examples/ballistic.py

Phase 1 has no viewer yet, so this is how you look at a trajectory: a printed
summary and a comparison against the closed-form solution with drag switched
off. Run it with drag on and off and compare the ranges — the difference is
larger than most people expect.
"""

from __future__ import annotations

import numpy as np

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.propulsion import Motor
from interceptor.core.state import EntityState
from interceptor.core.world import STANDARD_GRAVITY, World, WorldConfig
from interceptor.entities.missile import Missile
from interceptor.sim.engagement import ground_impact, run

LAUNCH_SPEED = 300.0
LAUNCH_ELEVATION_DEG = 45.0


def fly(*, with_drag: bool) -> None:
    elevation = np.radians(LAUNCH_ELEVATION_DEG)
    initial = EntityState(
        pos=np.array([0.0, 0.0, 0.1]),
        vel=np.array([0.0, LAUNCH_SPEED * np.cos(elevation), LAUNCH_SPEED * np.sin(elevation)]),
        mass=85.0,
    )

    world = World(WorldConfig(enable_drag=with_drag))
    world.add(
        Missile(
            "missile",
            initial,
            motor=Motor(),  # unpowered: pure ballistics
            aero=Aerodynamics(drag_coefficient=0.30, reference_area=0.02),
        )
    )

    result = run(world, duration=120.0, dt=1e-3, record_hz=100.0, stop=ground_impact("missile"))

    position = result.recorder.position("missile")
    speed = result.recorder.speed("missile")
    downrange = float(position[-1, 1])
    apex = float(position[:, 2].max())

    label = "with drag" if with_drag else "vacuum"
    print(f"\n{label:>10}  |  {result.reason}")
    print(f"{'flight time':>22}: {result.end_time:8.2f} s")
    print(f"{'downrange':>22}: {downrange:8.1f} m")
    print(f"{'apex':>22}: {apex:8.1f} m")
    print(f"{'impact speed':>22}: {speed[-1]:8.1f} m/s")

    if not with_drag:
        expected_range = LAUNCH_SPEED**2 * np.sin(2.0 * elevation) / STANDARD_GRAVITY
        expected_apex = (LAUNCH_SPEED * np.sin(elevation)) ** 2 / (2.0 * STANDARD_GRAVITY)
        print(f"{'analytic range':>22}: {expected_range:8.1f} m")
        print(f"{'analytic apex':>22}: {expected_apex:8.1f} m")


def main() -> None:
    print(f"Ballistic launch: {LAUNCH_SPEED:.0f} m/s at {LAUNCH_ELEVATION_DEG:.0f} degrees")
    fly(with_drag=False)
    fly(with_drag=True)
    print()


if __name__ == "__main__":
    main()
