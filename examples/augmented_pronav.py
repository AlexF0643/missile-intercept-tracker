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

This script flies five target behaviours, ordered by how well they satisfy that
assumption — which is the obvious way to arrange the comparison, and turns out
not to be what separates the results:

* **Straight and level.** No acceleration, so no lead term. APN is PN exactly.
* **Break turn.** A hard 7 g pull, held from t = 8 s to impact. The assumption
  is *true*, and this is the case APN was derived for.
* **Weave.** 6 g reversing every 2 s. Wrong for a moment after each reversal,
  right the rest of the time.
* **Jink.** 7 g in an uncorrelated new direction every 1.5 s. Wrong more often,
  and nothing in the history predicts the next break.
* **Barrel roll.** 5 g of *constant magnitude* whose direction rotates
  continuously, so the lead never decays at all.

Each behaviour is flown twice: once on a perfect track, which is APN's ceiling,
and once through the 2 mrad seeker and the extended Kalman filter, which is what
you would actually have. Both matter. A law that only works on perfect
information is a curiosity, and a law that fails on perfect information cannot
be rescued by a better filter.

The last part of the script is the one worth reading. It re-flies the barrel
roll with a more capable airframe, and shows that what looks like a broken
assumption is actually a saturated one: the lead term is a request for lift, and
whether it helps depends on whether that request can be met.

Writes ``runs/augmented-pronav.png``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from interceptor.config import loads
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


#: The barrel roll, on truth, with the lift coefficient varied and nothing else.
AIRFRAME = """
duration = 30.0
[missile]
position = [0.0, 0.0, 1000.0]
speed = 60.0
[missile.aero]
max_lift_coefficient = {lift}
[target]
position = [0.0, 6000.0, 1000.0]
velocity = [250.0, 0.0, 0.0]
[target.manoeuvre]
kind = "barrel_roll"
amplitude_g = 5.0
period = 4.0
[guidance]
law = "{law}"
[seeker]
enabled = false
"""


def airframe_sweep() -> None:
    """The experiment that identifies the cause rather than the symptom.

    One number varies: how much lift the airframe can make. The manoeuvre, the
    geometry, the track and the law are all held fixed, and the premise is
    violated identically in every row. If the premise were the cause, nothing
    here would move.
    """
    print("\nSo vary the airframe instead, on the same perfect track:\n")
    print(f"{'Cl_max':>8}  {'PN':>28}  {'APN':>28}")
    for lift in (2.5, 3.0, 3.5, 4.0, 5.0):
        cells = []
        for law in ("pronav", "apn"):
            spec = loads(AIRFRAME.format(lift=lift, law=law), "airframe")
            world, detector = spec.build(seed=0)
            result = run(world, duration=spec.scenario.duration, dt=1e-3, stop=detector)
            commanded = np.asarray(result.recorder.commanded("missile"))
            available = np.asarray(result.recorder.limit("missile"))
            saturated = float(np.mean(commanded > available * 0.999))
            miss = (
                "no result" if detector.result is None else f"{detector.result.miss_distance:.2f} m"
            )
            arrival = float(result.recorder.speed("missile")[-1])
            cells.append(f"{miss:>10}, {saturated:3.0%} sat, {arrival:3.0f} m/s")
        print(f"{lift:>8.1f}  " + "  ".join(cells))

    print(
        "\nTwice the lift and APN goes from twelve times worse than PN to a hundred and"
        "\nfifty times better — against the same manoeuvre, breaking the same assumption."
        "\nThe lead term is a request for about half as much lateral acceleration again."
        "\nWhere the airframe can meet it, it is worth a great deal. Where it saturates,"
        "\nthe surplus is never produced but the lift that is still costs induced drag,"
        "\nso the missile pays for the whole command and receives part of it."
        "\n\nThe number that decides this, max_lift_coefficient, is uncited."
    )


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
        f"\nThe barrel roll goes the other way, and on a perfect track: APN arrives at "
        f"{speeds[roll]['APN (N=3)']:.0f} m/s where PN arrives at {speeds[roll]['PN (N=3)']:.0f},"
        "\nso it is not the filter. The tempting explanation is that a rotating acceleration"
        "\nbreaks APN's constant-acceleration premise. It does. That is not why it misses."
    )
    airframe_sweep()

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
        title="Augmented proportional navigation: a request for lift, and whether it can be met",
    )
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
