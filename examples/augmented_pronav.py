"""What does knowing the target's acceleration buy, and what does it cost?

    python examples/augmented_pronav.py

Proportional navigation drives the line-of-sight rate to zero. That is exactly
right against a target flying straight, and always one step behind a target that
is accelerating: PN reacts to the bearing drift a manoeuvre has *already*
caused. Augmented proportional navigation adds a term for the acceleration
itself,

    a = N * V_c * (Omega x r_hat)  +  (N / 2) * a_t_perp

and the coefficient is not a tuning knob. It is the optimal-control solution for
a target holding *constant* acceleration, derived under the same criterion that
gives N = 3 against one holding none.

That assumption is the whole story, and this script is arranged to show both
sides of it. Five target behaviours, ordered by how well they satisfy it:

* **Straight and level.** No acceleration, so no lead term. APN is PN exactly.
* **Break turn.** A hard 7 g pull, held from t = 8 s to impact. The assumption
  is *true*, and this is the case APN was derived for.
* **Weave.** 6 g reversing every 2 s. The assumption is wrong but harmlessly so:
  the lead points the wrong way for a moment after each reversal, and is right
  the rest of the time.
* **Jink.** 7 g in an uncorrelated new direction every 1.5 s. Wrong more often,
  and nothing in the history predicts the next break.
* **Barrel roll.** 5 g of *constant magnitude* whose direction rotates
  continuously. The assumption is wrong in the one way that costs energy rather
  than merely failing to help — see the note printed at the end.

Each behaviour is flown twice: once on a perfect track, which is APN's ceiling,
and once through the 2 mrad seeker and the extended Kalman filter, which is what
you would actually have. Both matter. A law that only works on perfect
information is a curiosity, and a law that fails on perfect information cannot
be rescued by a better filter.

Writes ``runs/augmented-pronav.png``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from interceptor.entities.target import (
    Manoeuvre,
    barrel_roll,
    break_turn,
    jink,
    straight_and_level,
    weave,
)
from interceptor.guidance.base import GuidanceLaw
from interceptor.guidance.pronav import AugmentedProportionalNavigation, ProportionalNavigation
from interceptor.sensing.filters import ExtendedKalman
from interceptor.sensing.seeker import SeekerConfig
from interceptor.sim import scenarios
from interceptor.sim.engagement import run

OUTPUT = Path("runs")
SEEDS = 6
LETHAL_RADIUS = 5.0

#: Ordered by how well each satisfies APN's constant-acceleration premise.
BEHAVIOURS: dict[str, Manoeuvre] = {
    "Straight and level": straight_and_level(),
    "Break turn, 7 g at t=8 s": break_turn(amplitude_g=7.0, start_time=8.0),
    "Weave, 6 g / 4 s": weave(amplitude_g=6.0, period=4.0),
    "Jink, 7 g every 1.5 s": jink(amplitude_g=7.0, interval=1.5),
    "Barrel roll, 5 g / 4 s": barrel_roll(amplitude_g=5.0, period=4.0),
}

LAWS: dict[str, Callable[[], GuidanceLaw]] = {
    "PN (N=3)": lambda: ProportionalNavigation(3.0),
    "APN (N=3)": lambda: AugmentedProportionalNavigation(3.0),
}


def fly(
    manoeuvre: Manoeuvre,
    make_law: Callable[[], GuidanceLaw],
    *,
    seeker: bool,
    seed: int = 0,
) -> tuple[float | None, bool, float]:
    """One engagement. Returns miss distance, whether it hit, and terminal speed.

    Terminal speed is carried through because it is the evidence for *why* the
    barrel roll goes the way it does. A law that is merely aiming badly arrives
    at the same speed as one that is aiming well; a law paying for lift it
    cannot use arrives slow.
    """
    scenario = scenarios.crossing().with_manoeuvre(manoeuvre)
    world, detector = scenario.build(
        make_law(),
        seeker=SeekerConfig() if seeker else None,
        estimator=ExtendedKalman() if seeker else None,
        seed=seed,
    )
    result = run(world, duration=scenario.duration, dt=1e-3, stop=detector)
    speed = float(result.recorder.speed("missile")[-1])
    if detector.result is None:
        return None, False, speed
    return detector.result.miss_distance, detector.result.hit, speed


def main() -> None:
    width = max(len(name) for name in BEHAVIOURS) + 2

    # Perfect information first. This is the ceiling: whatever the filter does
    # afterwards, no track can be better than the truth, so a law that already
    # fails here is not going to be fixed downstream.
    print("Perfect information — the ceiling for each law")
    print(f"{'':>{width}}" + "".join(f"{name:>14}" for name in LAWS))
    speeds: dict[str, dict[str, float]] = {}
    for behaviour, manoeuvre in BEHAVIOURS.items():
        cells, row_speeds = [], {}
        for label, make in LAWS.items():
            miss, _, speed = fly(manoeuvre, make, seeker=False)
            cells.append("     no result" if miss is None else f"{miss:>12.2f} m")
            row_speeds[label] = speed
        speeds[behaviour] = row_speeds
        print(f"{behaviour:>{width}}" + "".join(cells))

    # Then the same thing through a real seeker and filter, several seeds each,
    # because a single seeded run of a noisy system is an anecdote.
    print(f"\nSeeker + EKF, {SEEDS} seeds — median miss and hits scored")
    print(f"{'':>{width}}" + "".join(f"{name:>20}" for name in LAWS))

    results: dict[str, dict[str, np.ndarray]] = {}
    scored: dict[str, dict[str, tuple[int, int]]] = {}
    for behaviour, manoeuvre in BEHAVIOURS.items():
        panel: dict[str, np.ndarray] = {}
        panel_hits: dict[str, tuple[int, int]] = {}
        cells = []
        for label, make in LAWS.items():
            misses, hits = [], 0
            for seed in range(SEEDS):
                miss, hit, _ = fly(manoeuvre, make, seeker=True, seed=seed)
                if miss is not None:
                    misses.append(miss)
                    hits += hit
            row = np.array(misses, dtype=np.float64)
            panel[label] = row
            panel_hits[label] = (hits, SEEDS)
            cells.append(f"{np.median(row):>11.2f} m{hits:>4}/{SEEDS}")
        results[behaviour] = panel
        scored[behaviour] = panel_hits
        print(f"{behaviour:>{width}}" + "".join(cells))

    roll = "Barrel roll, 5 g / 4 s"
    print(
        f"\nWhy the barrel roll goes the other way: on a perfect track APN arrives at "
        f"{speeds[roll]['APN (N=3)']:.0f} m/s where PN arrives at {speeds[roll]['PN (N=3)']:.0f}."
        "\nA rotating acceleration never decays, so neither does APN's lead — a standing"
        "\ncommand that points somewhere new every second, paying induced drag the whole"
        "\nway down for lift that buys nothing. It is the premise failing, not the filter."
    )

    try:
        from interceptor.viz.plots import save_estimator_comparison
    except ImportError:
        print("\n(install the viz extra for figures)")
        return

    path = save_estimator_comparison(
        results,
        OUTPUT / "augmented-pronav.png",
        hits=scored,
        lethal_radius=LETHAL_RADIUS,
        title="Augmented proportional navigation, where its assumption holds and where it does not",
    )
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
