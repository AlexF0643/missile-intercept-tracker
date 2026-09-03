# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 3 — proportional navigation. `a = N * V_c * (Omega x r_hat)` in vector
  form, projected perpendicular to the missile velocity (pure PN rather than
  true PN, since an aerodynamic missile can only generate force at right angles
  to the airflow). Plus a `plot_comparison` figure for flying several laws on
  the same scenario.
- Phase 3 exit criterion met by a wide margin: on the crossing geometry, PN
  cuts the miss distance from 40.757 m to 0.030 m — a factor of 1350 against
  the factor of 10 required — while using less peak acceleration than pure
  pursuit (7.8 g against 13.2 g).
- The comparison test asserts the *ratio* between the two laws rather than an
  absolute threshold, so it still fails if a change degrades both together.

- Phase 2 — the first intercepts, on perfect information. A `Track` carrying the
  engagement geometry and its derived quantities (range, closing speed, the
  line-of-sight rotation vector); a `TrackSource` interface with a truth
  implementation; the `GuidanceLaw` interface and a `PurePursuit` baseline; an
  `Autopilot` with an acceleration limiter and a 0.2 s first-order lag; guidance
  running at 100 Hz under the 1 kHz physics step with a zero-order hold; three
  standard scenarios (head-on, crossing, tail-chase); `ClosestApproach`, which
  finds the true point of closest approach by interpolating between physics
  steps rather than reading the smallest sample; and a five-panel matplotlib
  diagnostic figure behind an optional `viz` extra.
- Pure pursuit hits head-on (3.2 m) and in a tail chase (0.02 m), and misses a
  crossing target by 40.8 m while demanding 217 g from an airframe that can
  deliver 13. That failure is asserted in the test suite, so Phase 3's
  proportional navigation has a fixed baseline to beat.

### Changed

- The airframe's acceleration limit is now applied against the instantaneous
  state inside the equations of motion, not only at the guidance tick. A command
  clamped when issued could otherwise exceed the limit ten physics steps later,
  once the missile had slowed.

- Phase 1 — the world and ballistics. `EntityState` with flat-array packing for
  the integrator; pure `rk4` and `euler`; ENU and body-frame (FRD) conversions
  with exact round-trips; a `World` that steps entities at a fixed 1 ms step;
  exponential-atmosphere density; a two-phase `Motor` and point-mass
  `Aerodynamics` including the available-g envelope; a ballistic `Missile` and a
  kinematic `Target` with straight, weave and break-turn profiles; a multi-rate
  `Scheduler`, a preallocating `Recorder` and the `run` loop with termination
  conditions. 63 tests, including the Phase 1 exit criterion: a drag-free launch
  tracks the analytic parabola to 3.4e-10 m over 10 s.

- Phase 0 scaffolding: `src/` layout with hatchling packaging, Ruff lint and
  format, strict mypy, pytest with registered `gui` and `slow` markers, a
  GitHub Actions matrix across Python 3.11/3.12 on Ubuntu and Windows, and
  pre-commit hooks mirroring the CI checks.

[Unreleased]: https://github.com/AlexF0643/missile-intercept-tracker/compare/v0.1.0...HEAD
