"""Miss distance as a distribution, over two kinds of uncertainty.

Every result before this one was a median over six seeds. That is an anecdote
with error bars drawn on it, and it hides the thing an engagement study is
actually for: not how far the average round misses but what fraction of them
arrive close enough to matter, and what the ones that do not have in common.

**Two kinds of uncertainty, and only sampling one of them is misleading.**

*Aleatory* — the seeker's noise. Genuinely random, different every launch, and
what a conventional Monte Carlo samples. It gives a probability of kill with a
tight confidence interval.

*Epistemic* — the constants. Not random at all: the drag coefficient has one
true value and this project does not know it. Sampling only the noise produces a
beautifully precise number resting on a guess, and this project has already been
caught by exactly that. An uncited ``max_lift_coefficient`` was the difference
between augmented pronav being excellent and being catastrophic, and no amount
of extra seeds would have revealed it.

So the sampling is nested. Each *draw* fixes one plausible set of constants —
one candidate for what this missile actually is — and flies many *seeds* within
it. That gives a probability of kill per draw, with a binomial interval, and
then a spread of those probabilities across draws. The first number answers "how
reliable is this round"; the second answers "how much does that answer depend on
things nobody looked up". The second is usually the larger of the two, which is
the finding.

**Parallel because it has to be.** A thousand engagements is half an hour of one
core. Trials are independent, seeds are explicit, and nothing is shared, so a
process pool from the standard library divides that by however many cores are
present without any change to the physics. Results are keyed by draw and seed
rather than by arrival order, so the output does not depend on scheduling.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from interceptor import uncertainty
from interceptor.config import resolve
from interceptor.sim.engagement import run

__all__ = [
    "Study",
    "Trial",
    "fly_one",
    "study",
    "wilson_interval",
]

#: The same 1 kHz the rest of the project integrates at. A statistical study is
#: not a reason to quietly change the physics.
PHYSICS_STEP = 1e-3


@dataclass(frozen=True)
class Trial:
    """One engagement, reduced to the numbers worth keeping.

    Deliberately small and made of plain scalars: a thousand of these cross a
    process boundary, and a recorder holding two full trajectories would make
    that the expensive part of the study.
    """

    draw: int
    seed: int
    miss_distance: float
    hit: bool
    time: float
    arrival_speed: float
    #: Fraction of the flight where the guidance law asked for more lateral
    #: acceleration than the airframe could produce. The diagnostic that
    #: explained augmented pronav's failures, so it is worth carrying.
    saturated: float
    #: Seeker frames with no usable measurement, as a fraction of those tried.
    dropouts: float
    reached_closest_approach: bool


def fly_one(item: tuple[int, int, dict[str, Any]]) -> Trial:
    """Fly one engagement. A module-level function so a process pool can pickle it.

    Takes the draw index, the seed and an already-resolved scenario *table*
    rather than TOML text: the parameter draw has to modify the scenario anyway,
    and a nested dictionary of plain numbers crosses a process boundary as
    cheaply as a string does while skipping a parse per trial.
    """
    index, seed, scenario = item
    spec = resolve(scenario, "trial")
    world, detector = spec.build(seed=seed)
    result = run(world, duration=spec.scenario.duration, dt=PHYSICS_STEP, stop=detector)

    recorder = result.recorder
    commanded = np.asarray(recorder.commanded("missile"))
    available = np.asarray(recorder.limit("missile"))
    saturated = float(np.mean(commanded > available * 0.999)) if commanded.size else 0.0

    track = getattr(world["missile"], "track_source", None)
    attempts = max(len(recorder.time) - 1, 1)
    dropouts = float(getattr(track, "dropouts", 0)) / attempts

    intercept = detector.result
    return Trial(
        draw=index,
        seed=seed,
        # A run that never reached closest approach is a miss of unbounded size.
        # Recording the final separation rather than infinity keeps it plottable
        # and keeps the flag that says which it was.
        miss_distance=(
            intercept.miss_distance
            if intercept is not None
            else float(recorder.separation("missile", "target")[-1])
        ),
        hit=bool(intercept is not None and intercept.hit),
        time=float(intercept.time if intercept is not None else result.end_time),
        arrival_speed=float(recorder.speed("missile")[-1]),
        saturated=saturated,
        dropouts=dropouts,
        reached_closest_approach=intercept is not None,
    )


def wilson_interval(hits: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence interval for a proportion, by Wilson's method.

    Not ``p +/- z * sqrt(p(1-p)/n)``. The normal approximation is wrong in
    exactly the region this study cares about: at 60 hits from 60 it gives an
    interval of zero width, claiming certainty from a finite sample, and near
    zero it reaches below zero. Wilson's interval is barely longer to write,
    stays inside [0, 1] by construction, and behaves sensibly at the ends —
    60/60 becomes about [0.94, 1.0], which is an honest statement of what sixty
    successes can tell you.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = hits / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class Study:
    """Everything a Monte Carlo produced, and the questions it can answer."""

    trials: list[Trial]
    #: The constants used for each draw, by draw index. Draw 0 is always the
    #: nominal set, so "what the shipped scenario does" is always in the result.
    draws: list[dict[str, float]] = field(default_factory=list)
    name: str = ""

    @property
    def misses(self) -> np.ndarray:
        return np.array([t.miss_distance for t in self.trials], dtype=np.float64)

    @property
    def hit_fraction(self) -> float:
        """Probability of kill across every trial, noise and constants together."""
        return float(np.mean([t.hit for t in self.trials])) if self.trials else 0.0

    def by_draw(self) -> Iterator[tuple[int, list[Trial]]]:
        """Trials grouped by which set of constants they were flown under."""
        for index in sorted({t.draw for t in self.trials}):
            yield index, [t for t in self.trials if t.draw == index]

    def kill_probabilities(self) -> list[float]:
        """One probability of kill per draw — the epistemic spread."""
        return [float(np.mean([t.hit for t in group])) for _, group in self.by_draw()]

    def save(self, path: str | Path) -> Path:
        """Write the trials and the draws to a compressed archive.

        A study is half an hour of compute and a figure is a second of it, so
        the two should not be welded together. Anything that redraws — a
        different palette, an extra panel, a reviewer's question — reads this
        back instead of re-flying two thousand engagements.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.draws[0]) if self.draws else []
        np.savez_compressed(
            destination,
            name=np.array(self.name),
            fields=np.array([f.name for f in fields(Trial)]),
            trials=np.array(
                [[getattr(t, f.name) for f in fields(Trial)] for t in self.trials],
                dtype=np.float64,
            ),
            draw_keys=np.array(keys),
            draw_values=np.array([[d[k] for k in keys] for d in self.draws], dtype=np.float64),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Study:
        """Read back what :meth:`save` wrote."""
        with np.load(Path(path), allow_pickle=False) as archive:
            names = [str(n) for n in archive["fields"]]
            keys = [str(k) for k in archive["draw_keys"]]
            # Built by name rather than by position, so a field added to `Trial`
            # makes an old archive fail loudly here instead of silently
            # shifting every column along by one.
            column = {name: i for i, name in enumerate(names)}
            trials = [
                Trial(
                    draw=int(row[column["draw"]]),
                    seed=int(row[column["seed"]]),
                    miss_distance=float(row[column["miss_distance"]]),
                    hit=bool(row[column["hit"]]),
                    time=float(row[column["time"]]),
                    arrival_speed=float(row[column["arrival_speed"]]),
                    saturated=float(row[column["saturated"]]),
                    dropouts=float(row[column["dropouts"]]),
                    reached_closest_approach=bool(row[column["reached_closest_approach"]]),
                )
                for row in archive["trials"]
            ]
            draws = [dict(zip(keys, values, strict=True)) for values in archive["draw_values"]]
            return cls(trials=trials, draws=draws, name=str(archive["name"]))

    def nominal_trials(self) -> list[Trial]:
        """Just the trials flown with the shipped constants, i.e. draw 0."""
        return [t for t in self.trials if t.draw == 0]


def study(
    scenario: dict[str, Any],
    *,
    draws: int = 1,
    seeds: int = 100,
    workers: int | None = None,
    name: str = "",
    rng_seed: int = 20260906,
) -> Study:
    """Fly ``draws x seeds`` engagements and collect them.

    Args:
        scenario: A parsed scenario table, as :func:`interceptor.config.resolve`
            takes.
        draws: How many sets of constants to try. Draw 0 is always the nominal
            set, so ``draws=1`` is a conventional noise-only Monte Carlo and
            anything more adds the epistemic layer on top of it.
        seeds: Seeker noise seeds per draw.
        workers: Processes. Defaults to every core.
        rng_seed: Fixes which constants are drawn, so a study reproduces.

    Work is submitted as one flat list of trials rather than draw by draw, so a
    pool never sits idle waiting for a slow draw to finish — some parameter sets
    fly for twice as long as others, because a missile that cannot catch its
    target runs to the duration limit.
    """
    rng = np.random.default_rng(rng_seed)
    parameter_sets = [uncertainty.nominal()]
    parameter_sets += [uncertainty.draw(rng) for _ in range(max(draws - 1, 0))]

    work: list[tuple[int, int, dict[str, Any]]] = [
        (index, seed, uncertainty.apply(scenario, constants))
        for index, constants in enumerate(parameter_sets)
        for seed in range(seeds)
    ]

    count = workers if workers is not None else (os.cpu_count() or 1)
    trials = _fly_all(work, count)
    trials.sort(key=lambda t: (t.draw, t.seed))
    return Study(trials=trials, draws=parameter_sets, name=name)


def _fly_all(work: Sequence[tuple[int, int, dict[str, Any]]], workers: int) -> list[Trial]:
    """Run the trials, in parallel when that is worth doing."""
    if workers <= 1 or len(work) == 1:
        # One core, or one trial. A pool would cost more than it saved, and
        # staying in-process keeps a traceback readable while developing.
        return [fly_one(item) for item in work]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Chunked because the per-trial payload is small and the per-task
        # overhead is not: a thousand individual submissions spend real time in
        # the queue for work that takes two seconds each.
        return list(pool.map(fly_one, work, chunksize=max(1, len(work) // (workers * 8))))
