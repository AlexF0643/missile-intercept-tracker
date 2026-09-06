"""How reliable is this round, and how much does that answer depend on guesses?

    python examples/monte_carlo.py

Two questions, and the project has only ever been able to answer the first.

**How reliable is it?** Fly the same engagement many times with different seeker
noise and count how many arrive inside the lethal radius. That is a probability
of kill with a confidence interval, and it is what a Monte Carlo normally means.

**How much does that depend on things nobody looked up?** Every physical
constant in this model was chosen because it seemed plausible;
``interceptor/uncertainty.py`` now says so, one by one, and gives each a range
within which the truth plausibly lies. Drawing a set of constants from those
ranges gives one candidate for what this missile actually *is*. Flying seeds
inside each draw gives a probability of kill for that candidate. The spread of
those probabilities is the answer to the second question.

It matters because the project has already been caught by it once. Augmented
pronav missing a barrel roll by 469 m was written up as its constant-
acceleration premise failing; it was really an uncited lift coefficient, and
doubling that number turned the same engagement into a 2 cm intercept. No number
of extra noise seeds would ever have shown that.

**Two studies rather than one, and the reason is a measurement.** A first
attempt spent thirty seeds inside each of twenty draws and found that a draw's
probability of kill is almost always exactly 0 or exactly 1 — bimodal, with very
little in between. That is physically sensible: for a given airframe the missile
either has the energy and the lift to catch a weaving target or it does not, and
the seeker's noise then decides only the marginal cases. It also means seeds
spent inside a draw are largely wasted, because there is barely any variance in
there to measure. So the budget is split the way the variance actually lies:
many seeds at the shipped constants, to answer the first question precisely, and
many *draws* with few seeds each, to answer the second one at all.

Writes ``runs/monte-carlo.png`` and caches each study beside it, so that
redrawing does not mean re-flying.
"""

from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from interceptor import uncertainty
from interceptor.config import bundled_text
from interceptor.sim.montecarlo import Study, study, wilson_interval

OUTPUT = Path("runs")

#: The engagement APN was built for and PN now fails: a 6 g weave through a real
#: seeker. Anything easier makes both laws look identical; anything harder makes
#: both look hopeless.
SCENARIO = "crossing"
MANOEUVRE = {"kind": "weave", "amplitude_g": 6.0, "period": 4.0}

#: The conventional study: one set of constants, many seeds.
NOISE_SEEDS = 200

#: The honest one: many sets of constants, few seeds each. Ten is enough to see
#: which side of the bimodal split a draw fell on, which is all that is there.
DRAWS = 60
DRAW_SEEDS = 10

LAWS = (("PN (N=3)", "pronav"), ("APN (N=3)", "apn"))


def scenario_for(law: str) -> dict[str, Any]:
    base = tomllib.loads(bundled_text(SCENARIO))
    base["target"]["manoeuvre"] = dict(MANOEUVRE)
    base.setdefault("guidance", {})["law"] = law
    return base


def flown(law: str, tag: str, *, draws: int, seeds: int) -> Study:
    """One study, from cache when it is there. Delete runs/*.npz to re-fly."""
    cache = OUTPUT / f"monte-carlo-{law}-{tag}.npz"
    if cache.exists():
        result = Study.load(cache)
        print(f"  {len(result.trials)} engagements read from {cache}")
        return result

    started = time.perf_counter()
    result = study(scenario_for(law), draws=draws, seeds=seeds, name=law, rng_seed=20260906)
    result.save(cache)
    print(f"  {len(result.trials)} engagements in {time.perf_counter() - started:.0f} s")
    return result


def sensitivity(result: Study) -> list[tuple[str, float]]:
    """Which uncited constant moves the answer most.

    Rank correlation between each drawn constant and that draw's probability of
    kill. Rank rather than linear because the relationship is not expected to be
    straight — a missile either has the energy to arrive or it does not — and a
    monotone measure survives that where Pearson's would not.

    **Read this as a hint, not a result.** Thirteen constants vary at once, so
    each correlation is measured against the noise of the other twelve, and
    sixty draws is a small sample for that. Separating them properly means
    varying one at a time, which is a different and much longer study. What this
    can say is whether anything stands out; what it cannot say is a ranking to
    be trusted.
    """
    probabilities = np.array(result.kill_probabilities())
    ranked = _rank(probabilities)

    scores = []
    for constant in uncertainty.SAMPLED:
        values = np.array([draw[constant.path] for draw in result.draws])
        correlation = float(np.corrcoef(_rank(values), ranked)[0, 1])
        scores.append((constant.path, correlation))
    return sorted(scores, key=lambda pair: -abs(pair[1]))


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks, averaging ties — Spearman needs them and numpy has no rankdata."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # Averaging ties matters here more than usual: a probability of kill over
    # ten seeds takes eleven distinct values and lands on 0 or 1 most of the
    # time, so without this most of the sample would be ranked arbitrarily.
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def main() -> None:
    print(f"Scenario: {SCENARIO}, 6 g weave, through a 2 mrad seeker and an EKF.\n")

    noise: dict[str, Study] = {}
    constants: dict[str, Study] = {}
    for label, law in LAWS:
        print(f"{label} — noise only, {NOISE_SEEDS} seeds at the shipped constants")
        noise[label] = flown(law, "noise", draws=1, seeds=NOISE_SEEDS)
        print(f"{label} — {DRAWS} draws x {DRAW_SEEDS} seeds across plausible constants")
        constants[label] = flown(law, "draws", draws=DRAWS, seeds=DRAW_SEEDS)

    print("\n" + "=" * 74)
    print("What a conventional Monte Carlo would report")
    print("=" * 74)
    for label, _ in LAWS:
        trials = noise[label].nominal_trials()
        hits = sum(t.hit for t in trials)
        low, high = wilson_interval(hits, len(trials))
        misses = np.array([t.miss_distance for t in trials])
        print(
            f"  {label:<10} Pk {hits / len(trials):.3f}  [{low:.3f}, {high:.3f}]   "
            f"median miss {np.median(misses):6.2f} m   worst {misses.max():7.2f} m"
        )

    print("\n" + "=" * 74)
    print("What the project actually knows")
    print("=" * 74)
    for label, _ in LAWS:
        probabilities = np.array(constants[label].kill_probabilities())
        decided = float(np.mean((probabilities == 0.0) | (probabilities == 1.0)))
        print(
            f"  {label:<10} Pk {probabilities.min():.2f} to {probabilities.max():.2f} "
            f"across {len(probabilities)} plausible airframes; "
            f"mean {probabilities.mean():.2f}, median {np.median(probabilities):.2f}"
        )
        print(
            f"             {decided:.0%} of draws are all-or-nothing — "
            "the airframe decides, not the noise"
        )

    influence = sensitivity(constants["APN (N=3)"])
    print("\nWhich guess moves the answer? (rank correlation with Pk, APN — indicative only)")
    for path, score in influence[:6]:
        print(f"  {score:+.2f}  {path}")

    print(
        "\nThe interval in the first table is what a Monte Carlo usually reports."
        "\nThe range in the second is what this model can actually support."
    )

    try:
        from interceptor.viz.plots import save_monte_carlo
    except ImportError:
        print("\n(install the viz extra for figures)")
        return

    # The distribution panel wants the many-seeded study and the spread panel
    # wants the many-drawn one, so each is handed the sample it belongs to:
    # draw 0 from the noise study, every draw from the other.
    combined = {
        label: Study(
            trials=noise[label].nominal_trials()
            + [t for t in constants[label].trials if t.draw > 0],
            draws=constants[label].draws,
            name=label,
        )
        for label, _ in LAWS
    }
    path = save_monte_carlo(
        combined,
        influence,
        OUTPUT / "monte-carlo.png",
        lethal_radius=5.0,
        title="Probability of kill, and how much of it rests on guesses",
    )
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
