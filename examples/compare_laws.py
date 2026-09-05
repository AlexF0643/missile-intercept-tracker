"""Pure pursuit against proportional navigation, on identical scenarios.

    python examples/compare_laws.py

Same missile, same target, same truth data — only the guidance law differs.
Figures go to ``runs/``; needs ``pip install -e ".[viz]"``.

The crossing geometries are the ones to look at. Pure pursuit steers at where
the target is and ends up in a tail chase; proportional navigation steers to
stop the *bearing* drifting and flies an almost straight line to a point ahead
of the target.

Read the time and arrival columns, not only the miss distance. Against a target
obliging enough to fly straight, pursuit does eventually arrive — it runs the
target down over nearly twenty seconds and gets there with a quarter of the
closing speed, its energy spent turning. Miss distance cannot tell that outcome
apart from a clean intercept, which is why this table reports four numbers
rather than one.

Give the target a weave and the distinction stops being subtle: pursuit misses
by 89 m where PN misses by 6. And PN does it using *less* acceleration on the
straight case, which is the part that surprises people.
"""

from __future__ import annotations

from pathlib import Path

from interceptor.entities.target import weave
from interceptor.guidance.base import GuidanceLaw
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
GRAVITY = 9.80665


def _slug(name: str) -> str:
    """A filename from a human label. Spaces and commas do not belong in one."""
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-").replace("--", "-")


def main() -> None:
    laws: list[GuidanceLaw] = [PurePursuit(gain=4.0), ProportionalNavigation(3.0)]

    cases = [
        ("head-on", scenarios.head_on, None),
        ("crossing", scenarios.crossing, None),
        ("crossing, weaving 6 g", scenarios.crossing, weave(6.0, 4.0)),
        ("tail-chase", scenarios.tail_chase, None),
    ]

    for name, factory, manoeuvre in cases:
        print(f"\n{name}")
        print(
            f"  {'law':<18}{'miss (m)':>10}{'at':>8}{'closing':>10}"
            f"{'peak demand':>14}{'peak used':>12}{'arrival':>11}"
        )

        finished = []
        for law in laws:
            scenario = factory()
            if manoeuvre is not None:
                scenario = scenario.with_manoeuvre(manoeuvre)
            world, detector = scenario.build(law)
            result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
            record = result.recorder
            outcome = detector.result

            miss = "never closed" if outcome is None else f"{outcome.miss_distance:10.3f}"
            when = "     -  " if outcome is None else f"{outcome.time:7.2f}s"
            closing = "        - " if outcome is None else f"{outcome.closing_speed:8.0f}  "
            print(
                f"  {law.name:<18}{miss:>10}{when:>8}{closing:>10}"
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
            OUTPUT / f"comparison-{_slug(name)}.png",
            title=f"Pure pursuit vs proportional navigation · {name}",
        )
        print(f"  figure: {path}")

    print()


if __name__ == "__main__":
    main()
