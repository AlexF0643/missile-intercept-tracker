"""What a real seeker does to a guidance law that was perfect on truth data.

    python examples/seeker_sweep.py

Sweeps the seeker's angle noise from nothing to a realistic 2 mrad, flying the
crossing engagement several times at each level, and writes the trend to
``runs/seeker-noise-sweep.png``. Also writes a diagnostic figure of one fully
degraded run, for comparison against the Phase 3 one.

The result to expect: proportional navigation intercepts within a couple of
metres up to about 0.2 mrad of angle noise, and comes apart completely by 2.

Nothing about the guidance law has changed. What changed is that the relative
velocity it needs is now obtained by differencing two noisy positions ten
milliseconds apart, which multiplies the measurement error by a hundred. Phase 5
puts an estimator in that gap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.sensing.seeker import SeekerConfig
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
FACTORS = (0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = 8


def fly(config: SeekerConfig | None, seed: int = 0) -> tuple[float | None, object]:
    scenario = scenarios.crossing()
    world, detector = scenario.build(ProportionalNavigation(3.0), seeker=config, seed=seed)
    result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    miss = None if detector.result is None else detector.result.miss_distance
    return miss, (result, detector.result)


def main() -> None:
    base = SeekerConfig()

    baseline, _ = fly(None)
    print(f"Perfect information (Phase 3 baseline): {baseline:.3f} m\n")

    print(f"{'angle sigma':>13}{'median miss':>14}{'worst':>10}{'hits':>8}")
    sigmas: list[float] = []
    grid: list[list[float]] = []

    for factor in FACTORS:
        config = base.scaled(factor)
        misses = []
        hits = 0
        for seed in range(SEEDS):
            miss, (_, intercept) = fly(config, seed)
            if miss is not None:
                misses.append(miss)
                hits += bool(intercept and intercept.hit)  # type: ignore[union-attr]
        if not misses:
            continue
        sigmas.append(config.angle_sigma * 1e3)
        grid.append(misses)
        row = np.array(misses)
        print(
            f"{config.angle_sigma * 1e3:>10.3f} mr{np.median(row):>13.2f}m"
            f"{row.max():>9.1f}m{hits:>5}/{SEEDS}"
        )

    try:
        from interceptor.viz.plots import save_engagement, save_noise_sweep
    except ImportError:
        print("\n(install the viz extra for figures)")
        return

    path = save_noise_sweep(
        sigmas,
        np.array(grid),
        OUTPUT / "seeker-noise-sweep.png",
        lethal_radius=5.0,
        baseline=baseline,
        title="Proportional navigation degrading with seeker angle noise",
    )
    print(f"\nfigure: {path}")

    _, (result, intercept) = fly(base, seed=0)
    path = save_engagement(
        result,  # type: ignore[arg-type]
        OUTPUT / "seeker-degraded.png",
        intercept=intercept,  # type: ignore[arg-type]
        title="ProNav with a realistic seeker, no estimator",
    )
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
