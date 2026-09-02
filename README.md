# Missile Intercept Tracker

[![CI](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexF0643/missile-intercept-tracker/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A 3-DOF missile intercept simulation in Python. A noisy radar seeker measures a
manoeuvring target a hundred times a second, a filter turns those measurements
into a track, and a proportional-navigation law turns that track into steering
commands — rendered live in 3D.

> **Status: Phase 1.** World, integrator and ballistics. No guidance yet.

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
| 1 | World, RK4 integrator, ballistics | Drag-free launch matches the analytic parabola to 1e-6 over 10 s | ⬜ |
| 2 | Pure pursuit on truth data | First intercept against a non-manoeuvring target | ⬜ |
| 3 | Proportional navigation on truth data | ≥10× lower miss distance than pursuit on a crossing target | ⬜ |
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
pytest                    # full suite
pytest -m "not gui"       # what CI runs
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
