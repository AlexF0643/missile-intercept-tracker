"""Fly pure pursuit against all three standard geometries.

    python examples/pursuit.py

Prints the outcome of each and writes a diagnostic figure per scenario into
``runs/``. Requires the plotting extra::

    pip install -e ".[viz]"

What to look for. Pure pursuit hits comfortably head-on and in a tail chase,
and misses badly on the crossing geometry — because it steers at where the
target *is*, and against a crossing target that point is always behind where
the target will be. Watch the acceleration panel: the commanded trace runs away
to hundreds of g in the last second while the achieved trace stays pinned to the
airframe limit. That gap is the failure, and no amount of gain fixes it.

Phase 3 replaces this law with proportional navigation, which steers at where
the target is *going*, and the same three figures look completely different.
"""

from __future__ import annotations

from pathlib import Path

from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
GRAVITY = 9.80665


def main() -> None:
    law = PurePursuit(gain=4.0)
    print(f"Guidance law: {law.name} (gain {law.gain})\n")

    for factory in (scenarios.head_on, scenarios.crossing, scenarios.tail_chase):
        scenario = factory()
        world, detector = scenario.build(law)
        result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)

        record = result.recorder
        outcome = detector.result
        print(f"{scenario.name:>11}: {outcome if outcome is not None else 'never closed'}")
        print(
            f"{'':>11}  peak demand {record.commanded('missile').max() / GRAVITY:6.1f} g"
            f"  ·  peak achieved {record.achieved('missile').max() / GRAVITY:5.1f} g"
            f"  ·  arrival speed {record.speed('missile')[-1]:4.0f} m/s"
        )

        try:
            from interceptor.viz.plots import save_engagement
        except ImportError:
            continue

        path = save_engagement(
            result,
            OUTPUT / f"pursuit-{scenario.name}.png",
            intercept=outcome,
            title=f"Pure pursuit · {scenario.name}",
        )
        print(f"{'':>11}  figure: {path}")

    print()


if __name__ == "__main__":
    main()
