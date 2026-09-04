# Missile Intercept Tracker

[![CI](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A 3-DOF missile intercept simulation in Python. A noisy radar seeker measures a
manoeuvring target a hundred times a second, a filter turns those measurements
into a track, and a proportional-navigation law turns that track into steering
commands — rendered live in 3D.

> **Status: Phase 5.** A filter between the seeker and the guidance law turns a
> 1.5 km miss into a hit.

![Miss distance by estimator against three target behaviours](docs/assets/estimator-comparison.png)

Phase 4 ended in failure, deliberately left in place: proportional navigation
that intercepted within 3 cm on perfect information missed by **1577 m** once it
had to work from a 2 mrad seeker. Nothing was wrong with the guidance law. What
was wrong is that relative velocity was obtained by differencing two noisy
positions 10 ms apart, which multiplies the angle error by a hundred.

Phase 5 puts an estimator in that gap — the same seeker, the same guidance law,
something sensible in between. **1577 m becomes 0.88 m**, a factor of about
1700, and the extended Kalman filter scores 6 hits from 6 against all three
target behaviours.

The more interesting result is the one the three panels exist to show. Against a
straight target the heavily-smoothed alpha-beta filter is the *best* thing here
(0.40 m): averaging beats noise, and there is no signal being averaged away.
Against a 7 g break turn the same filter is the *only* one that fails outright —
9.58 m, 0 hits from 6 — because the smoothing that rejected the noise also
rejects the manoeuvre. Its gains were fixed in advance, and it cannot revisit
that decision when the target does something new.

The EKF is never quite the best in any single panel and never bad in any of
them, because it recomputes how much to trust its own prediction on every frame
from its own covariance. That is the trade the phase is about: a fixed gain has
to be chosen for behaviour you do not get to know beforehand.

The filter also **coasts through a 0.5 s blackout** — it keeps predicting when
there is nothing to correct with, where the naive version threw the track away
and the missile flew blind.

### Is the filter honest about its own uncertainty?

Miss distance says whether the estimate was *accurate*. It says nothing about
whether the covariance the filter reports alongside it is *truthful*, and those
come apart in a way that matters. A filter that understates its uncertainty runs
too small a gain, so it discounts measurements that disagree with it — tracking
a quiet target immaculately and then arriving late on the manoeuvre that counts.

The standard check is NEES: measure the error in units of the filter's own
claimed uncertainty, and it should average to the number of states, here 6.
Across 24 independent seeds the EKF scores **6.16 against an expected 6.00**,
inside the 95% interval, and stays there against a 6 g weave its
constant-acceleration model does not describe and across a 600-fold range of
process-noise tuning.

It did not the first time it was asked. It scored **1804** — a three-sigma bias
hiding under a filter that was hitting the target anyway. The seeker models one
frame of processing latency and stamps each measurement with when it was
*taken*, but the track was correcting its current state with that stale
measurement and with the missile's *current* position. That folds one frame of
relative motion into every estimate as a standing offset: 6.5 m at 650 m/s of
closing, against a filter claiming about 2 m of uncertainty.

Fixing it barely moved the miss distance — 0.94 m to 0.88 m — which is precisely
why it needed a consistency check to find. It would have quietly corrupted the
Phase 7 miss-distance distribution, where the covariance stops being diagnostic
and starts being the answer.

![Pure pursuit against proportional navigation on a crossing target](docs/assets/comparison-crossing.png)

Same missile, same target, same data — only the guidance law differs. Pure
pursuit steers at where the target *is* and ends up in a tail chase, missing by
**40.8 m**. Proportional navigation steers to stop the *bearing* drifting, flies
inside that arc to a point ahead of the target, and misses by **0.03 m**.

And it does so using less acceleration, not more: PN spends 7.8 g early to set
up the geometry and then coasts at 3.5 g, while pursuit ramps to 13 g in the
last two seconds and still arrives behind.

```bash
python examples/estimator_comparison.py   # the figure above (~7 min)
python examples/seeker_sweep.py           # miss distance against seeker noise
python examples/compare_laws.py           # pursuit vs PN, all three geometries
python examples/pursuit.py                # one law, five diagnostic panels
python examples/ballistic.py              # unguided flight, vacuum vs drag
```

```
crossing geometry, realistic seeker, 6 seeds       median miss    hits
  no estimator (Phase 4)                              1577.2 m    0/6
  alpha-beta, a=0.05, straight target                    0.40 m   6/6
  alpha-beta, a=0.05, 7 g break turn                     9.58 m   0/6
  EKF, jerk 60, straight target                          0.88 m   6/6
  EKF, jerk 60, 7 g break turn                           2.98 m   6/6
  perfect information (Phase 3 baseline)                 0.03 m   6/6
```

```
crossing
  law                 miss (m)   peak demand   peak used    arrival
  PurePursuit           40.757       217.4 g      13.2 g      535 m/s
  ProNav (N=3)           0.030       816.4 g       7.8 g      531 m/s
```

## Why this exists

The interesting behaviour in an engagement simulation lives at the boundaries
between three layers, not inside any one of them:

- **World** — truth. Where everything actually is, integrated at a fixed step.
- **Seeker** — the missile's only, deliberately lossy, window onto that truth:
  noisy, delayed, limited in field of view and prone to dropping out.
- **Guidance** — estimates what it can from that window, and commands a lateral
  acceleration the airframe may or may not be able to deliver.

A guidance law that intercepts perfectly on truth data starts missing when fed a
2 mrad angle error; a filter buys the performance back; the filter then lags on a
hard-manoeuvring target and needs a target-acceleration term. That progression is
the project.

Each layer talks to the next through one small type. The guidance law is handed
a `Track` and nothing else, so the same proportional navigation runs unchanged on
truth, on raw seeker measurements, and behind a filter — which makes "try this
law on perfect information" a one-line move for diagnosing anything downstream of
it.

## Roadmap

| Phase | Scope | Exit criterion | Status |
| :---- | :---- | :------------- | :----- |
| 0 | Scaffolding, lint, types, CI | Green CI on an empty project | ✅ |
| 1 | World, RK4 integrator, ballistics | Drag-free launch matches the analytic parabola to 1e-6 over 10 s | ✅ 3.4e-10 m |
| 2 | Pure pursuit on truth data | First intercept against a non-manoeuvring target | ✅ 3.2 m head-on |
| 3 | Proportional navigation on truth data | ≥10× lower miss distance than pursuit on a crossing target | ✅ 1350× |
| 4 | Seeker: frames, gimbal gating, noise, dropouts | Monotonic noise-vs-miss-distance sweep | ✅ 0.02 m → 1577 m |
| 5 | Estimation: alpha-beta, then EKF | Inside the 5 m lethal radius on a realistic seeker; survives a 0.5 s dropout | ✅ 1577 m → 0.88 m |
| 6 | Real-time 3D viewer | A recording good enough to head this README | ⬜ |
| 7 | Manoeuvring targets, augmented PN, Monte Carlo | Miss-distance distribution over 1000 runs | ⬜ |

Phase 5's exit criterion was originally written as *"within 2× of the
perfect-information baseline"* — 0.06 m. That criterion was wrong, and it is
worth saying why rather than quietly restating it. Perfect information has no
glint; a real seeker sees a target as a scattering body roughly 1.5 m across,
and which part of it dominates the return wanders from pulse to pulse. As range
falls that metre-scale wander subtends a *growing* angle, so the last second of
flight is the noisiest. No filter can average away an error that peaks exactly
when there is no time left to average. Sub-metre is the floor the physics
allows, and the criterion was changed to the one that means something: does the
round arrive inside its lethal radius.

## Install

```bash
git clone https://github.com/AlexF0643/missile-intercept-tracker.git
cd missile-intercept-tracker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Develop

```bash
pip install -e ".[dev,viz]"   # viz adds matplotlib for the diagnostic figures
pytest                        # full suite
pytest -m "not gui"           # what CI runs
ruff check . && ruff format .
mypy
```

## Modelling assumptions

Stated up front, because the boundary of a model is part of the model:

- 3 degrees of freedom — point mass, no attitude state. The seeker boresight is
  taken as the velocity vector, i.e. zero angle of attack.
- Flat, non-rotating earth. No Coriolis, no earth curvature.
- Exponential atmosphere, `rho = 1.225 * exp(-h / 8500)`.
- Point-mass aerodynamics: a drag coefficient and a reference area, no full
  aerodynamic database.
- The seeker is modelled at the measurement level — true geometry corrupted by
  noise, latency and dropouts — not at the level of transmitted waveforms.

## References

- Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA.
- Siouris, G. M. *Missile Guidance and Control Systems*, Springer.

## License

MIT — see [LICENSE](LICENSE).
