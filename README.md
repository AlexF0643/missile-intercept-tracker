# Missile Intercept Tracker

[![CI](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A 3-DOF missile intercept simulation in Python. A noisy radar seeker measures a
manoeuvring target a hundred times a second, a filter turns those measurements
into a track, and a proportional-navigation law turns that track into steering
commands — rendered live in 3D.

> **Status: Phase 3.** Proportional navigation, flown on perfect information.

![Pure pursuit against proportional navigation on a crossing target](docs/assets/comparison-crossing.png)

Same missile, same target, same data — only the guidance law differs. Pure
pursuit steers at where the target *is* and ends up in a tail chase, missing by
**40.8 m**. Proportional navigation steers to stop the *bearing* drifting, flies
inside that arc to a point ahead of the target, and misses by **0.03 m**.

And it does so using less acceleration, not more: PN spends 7.8 g early to set
up the geometry and then coasts at 3.5 g, while pursuit ramps to 13 g in the
last two seconds and still arrives behind.

```bash
python examples/compare_laws.py   # the figure above, for all three geometries
python examples/pursuit.py        # one law, five diagnostic panels
python examples/ballistic.py      # unguided flight, vacuum vs drag
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

## Roadmap

| Phase | Scope | Exit criterion | Status |
| :---- | :---- | :------------- | :----- |
| 0 | Scaffolding, lint, types, CI | Green CI on an empty project | ✅ |
| 1 | World, RK4 integrator, ballistics | Drag-free launch matches the analytic parabola to 1e-6 over 10 s | ✅ 3.4e-10 m |
| 2 | Pure pursuit on truth data | First intercept against a non-manoeuvring target | ✅ 3.2 m head-on |
| 3 | Proportional navigation on truth data | ≥10× lower miss distance than pursuit on a crossing target | ✅ 1350× |
| 4 | Seeker: frames, FOV/gimbal gating, noise, dropouts | Monotonic noise-vs-miss-distance sweep | ⬜ |
| 5 | Estimation: alpha-beta, then EKF | Within 2× of the perfect-information baseline; survives a 0.5 s dropout | ⬜ |
| 6 | Real-time 3D viewer | A recording good enough to head this README | ⬜ |
| 7 | Manoeuvring targets, augmented PN, Monte Carlo | Miss-distance distribution over 1000 runs | ⬜ |

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
