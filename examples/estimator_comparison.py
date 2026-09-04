"""Does a filter rescue proportional navigation from a realistic seeker?

    python examples/estimator_comparison.py

Phase 4 ended with a failure that was left in place on purpose: the same
proportional navigation that intercepted within 3 cm on perfect information
missed by 1.5 km once it had to work from a 2 mrad seeker, because the relative
velocity it needs was obtained by differencing two noisy positions 10 ms apart.
That multiplies the angle error by a hundred.

This script puts an estimator in that gap and flies the crossing engagement
against three target behaviours, several seeds each, writing the result to
``runs/estimator-comparison.png``.

Three behaviours, because the interesting finding only appears when they are
compared. A filter is a bet on how much the target is about to surprise you:

* **Straight and level.** No surprises. The heavier the smoothing, the better —
  averaging beats noise and there is no signal being averaged away.
* **Weave.** A continuous 6 g oscillation. Smoothing now removes some of the
  thing you are trying to track.
* **Break turn.** Nothing, then a hard 7 g turn at t = 8 s. The worst case for
  a filter tuned on the assumption that yesterday predicts today.

The alpha-beta filter has fixed gains chosen in advance. The extended Kalman
filter chooses its gains from its own uncertainty, every frame. Watch which one
wins each panel, and then which one you would actually load onto a missile that
has to fly against a target whose behaviour you do not get to know beforehand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from interceptor.entities.target import Manoeuvre, break_turn, straight_and_level, weave
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.sensing.filters import AlphaBeta, Estimator, ExtendedKalman
from interceptor.sensing.seeker import SeekerConfig
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
SEEDS = 6
LETHAL_RADIUS = 5.0

#: Target behaviours, easiest first.
BEHAVIOURS: dict[str, Manoeuvre] = {
    "Straight and level": straight_and_level(),
    "Weave, 6 g / 4 s": weave(amplitude_g=6.0, period=4.0),
    "Break turn, 7 g at t=8 s": break_turn(amplitude_g=7.0, start_time=8.0),
}

#: The estimators under test. The alpha-beta entries bracket the trade
#: deliberately: alpha = 0.05 is heavy smoothing, alpha = 0.2 is light. The EKF
#: entries vary only the assumed jerk — how much the filter believes the target
#: can change its acceleration — which is the single knob that decides how
#: readily it abandons its own prediction.
ESTIMATORS: dict[str, Callable[[], Estimator]] = {
    "alpha-beta, a=0.05": lambda: AlphaBeta(alpha=0.05),
    "alpha-beta, a=0.20": lambda: AlphaBeta(alpha=0.20),
    "EKF, jerk 20": lambda: ExtendedKalman(jerk_sigma=20.0),
    "EKF, jerk 60": lambda: ExtendedKalman(jerk_sigma=60.0),
    "EKF, jerk 150": lambda: ExtendedKalman(jerk_sigma=150.0),
}


def fly(
    manoeuvre: Manoeuvre,
    make_estimator: Callable[[], Estimator] | None,
    seed: int,
) -> tuple[float | None, bool]:
    """One engagement. Returns the miss distance and whether it was a hit.

    The estimator arrives as a factory rather than an instance because a filter
    carries state — its estimate and its covariance — and reusing one across
    runs would let the second run start out already convinced of where a
    different target was.
    """
    scenario = scenarios.crossing().with_manoeuvre(manoeuvre)
    world, detector = scenario.build(
        ProportionalNavigation(3.0),
        seeker=SeekerConfig(),
        estimator=None if make_estimator is None else make_estimator(),
        seed=seed,
    )
    run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    if detector.result is None:
        return None, False
    return detector.result.miss_distance, detector.result.hit


def perfect_information(manoeuvre: Manoeuvre) -> float:
    """The Phase 3 baseline: the same law with a flawless track."""
    scenario = scenarios.crossing().with_manoeuvre(manoeuvre)
    world, detector = scenario.build(ProportionalNavigation(3.0))
    run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    return 0.0 if detector.result is None else detector.result.miss_distance


def main() -> None:
    baseline = perfect_information(straight_and_level())
    print(f"Perfect information, straight target: {baseline:.3f} m")

    # The Phase 4 failure, re-measured rather than quoted, so the comparison is
    # against this build of the code and not against a number in a changelog.
    naive = [fly(straight_and_level(), None, seed)[0] for seed in range(SEEDS)]
    clean = [m for m in naive if m is not None]
    print(f"Seeker, no estimator (Phase 4):      {np.median(clean):.1f} m\n")

    results: dict[str, dict[str, np.ndarray]] = {}
    scored: dict[str, dict[str, tuple[int, int]]] = {}

    header = f"{'estimator':>22}" + "".join(f"{name:>28}" for name in BEHAVIOURS)
    print(header)
    rows: dict[str, list[str]] = {name: [] for name in ESTIMATORS}

    for behaviour, manoeuvre in BEHAVIOURS.items():
        panel: dict[str, np.ndarray] = {}
        panel_hits: dict[str, tuple[int, int]] = {}
        for label, make in ESTIMATORS.items():
            misses = []
            hits = 0
            for seed in range(SEEDS):
                miss, hit = fly(manoeuvre, make, seed)
                if miss is not None:
                    misses.append(miss)
                    hits += hit
            row = np.array(misses, dtype=np.float64)
            panel[label] = row
            panel_hits[label] = (hits, SEEDS)
            rows[label].append(f"{np.median(row):>16.2f} m{hits:>4}/{SEEDS} hits")
        results[behaviour] = panel
        scored[behaviour] = panel_hits

    for label, cells in rows.items():
        print(f"{label:>22}" + "".join(cells))

    try:
        from interceptor.viz.plots import save_estimator_comparison
    except ImportError:
        print("\n(install the viz extra for figures)")
        return

    path = save_estimator_comparison(
        results,
        OUTPUT / "estimator-comparison.png",
        hits=scored,
        lethal_radius=LETHAL_RADIUS,
        baseline=baseline,
        title="Estimator choice against three target behaviours",
    )
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
