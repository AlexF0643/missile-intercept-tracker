"""Pure pursuit against proportional navigation, on identical scenarios.

    python examples/compare_laws.py

Same missile, same target, same truth data — only the guidance law differs.
Figures go to ``runs/``; needs ``pip install -e ".[viz]"``.

The crossing geometry is the one to look at. Pure pursuit steers at where the
target is and ends up in a tail chase; proportional navigation steers to stop
the *bearing* drifting and flies an almost straight line to a point ahead of the
target. The miss distance drops by three orders of magnitude, and — the part
that surprises people — PN gets there using less acceleration, not more.
"""

from __future__ import annotations

from pathlib import Path

from interceptor.guidance.base import GuidanceLaw
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
GRAVITY = 9.80665


def main() -> None:
    laws: list[GuidanceLaw] = [PurePursuit(gain=4.0), ProportionalNavigation(3.0)]

    for factory in (scenarios.head_on, scenarios.crossing, scenarios.tail_chase):
        name = factory().name
        print(f"\n{name}")
        print(f"  {'law':<18}{'miss (m)':>10}{'peak demand':>14}{'peak used':>12}{'arrival':>11}")

        finished = []
        for law in laws:
            scenario = factory()
            world, detector = scenario.build(law)
            result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
            record = result.recorder
            outcome = detector.result

            miss = "never closed" if outcome is None else f"{outcome.miss_distance:10.3f}"
            print(
                f"  {law.name:<18}{miss:>10}"
                f"{record.commanded('missile').max() / GRAVITY:12.1f} g"
                f"{record.achieved('missile').max() / GRAVITY:10.1f} g"
                f"{record.speed('missile')[-1]:9.0f} m/s"
            )
            finished.append((law.name, result, outcome))

        try:
            from interceptor.viz.plots import LawRun, save_comparison
        except ImportError:
            continue

        path = save_comparison(
            [
                LawRun(label=label, result=result, intercept=outcome)
                for label, result, outcome in finished
            ],
            OUTPUT / f"comparison-{name}.png",
            title=f"Pure pursuit vs proportional navigation · {name}",
        )
        print(f"  figure: {path}")

    print()


if __name__ == "__main__":
    main()
