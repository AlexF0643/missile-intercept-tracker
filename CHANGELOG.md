# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 6 — the engagement can be watched. `interceptor.viz.scene` describes
  what to draw at each instant as plain numpy: positions, cumulative trails, the
  sightline, and the derived readout figures. Two renderers consume it.
  `interceptor.viz.record` writes an animated GIF or MP4 with matplotlib;
  `interceptor.viz.live` opens an orbitable WebGL scene through VPython. Neither
  contains geometry of its own, so they cannot disagree about where anything is,
  and all the arithmetic sits in the one module a test can reach — a live 3D
  window cannot be asserted about on a build server, so nothing that matters was
  left inside one.
- `interceptor record` and `interceptor view` fly a scenario and show it. The
  endgame plays at quarter speed by default, because at Mach 2 the last hundred
  metres pass in under a tenth of a second — three frames at 30 fps of the only
  part anybody watches for.
- Phase 6 exit criterion met: a recording good enough to head the README, which
  is now what heads it.
- VPython is a new optional extra, `live`, deliberately separate from `viz`. It
  starts a web server and opens a browser tab when imported, so
  `interceptor.viz.live` imports it inside a function and reports its absence as
  an instruction rather than a traceback. A test asserts that importing the
  module does not pull VPython in, because if that regresses every headless
  Monte Carlo run pays for a viewer nobody asked for.
- The vertical scale of a recording is exaggerated and the factor is written on
  the axis. An air-to-air engagement is 6 km across and 150 m tall; drawn to a
  true cube it is a smear through empty sky. The distortion is necessary, so it
  is labelled rather than hidden.
- `save_flight` chooses its writer before building the animation. Asking for an
  `.mp4` on a machine without ffmpeg raised the right error, but only after
  constructing a `FuncAnimation` that was then collected unrendered — which
  matplotlib warns about from a destructor, so the warning surfaced as an
  unraisable exception attributed to whichever unrelated test the garbage
  collector happened to interrupt. Nothing is now built until there is something
  to write with.
- The test for that path fakes ffmpeg's absence instead of skipping when ffmpeg
  is present. It previously skipped on the machine it was written on, so it never
  ran there and the bug above shipped. A test that only runs on other people's
  machines is not a test.
- The camera is fixed by default when recording. Parallax helps a projected 3D
  path read as depth, but a moving camera changes every background pixel on
  every frame and a GIF stores frames as differences — the same animation is
  1.2 MB with a fixed camera and 3.5 MB with a drifting one. Orbiting is what
  the live viewer is for, with a mouse. `--orbit` turns it on anyway.
- Phase 5 — estimation. `interceptor.sensing.filters` supplies two estimators
  behind a common `Estimator` interface that splits `predict` from `correct`,
  so a dropout costs the track confidence rather than the track itself.
  `AlphaBeta` runs fixed gains with `beta = alpha^2 / (2 - alpha)` for a
  critically damped response. `ExtendedKalman` carries a 9-state
  `[position, velocity, acceleration]` estimate of the target's *absolute*
  motion — absolute rather than relative, so the missile's own acceleration
  never has to appear in the process model — with a constant-acceleration
  transition, white-noise-jerk process noise, a numerical Jacobian of the
  range/azimuth/elevation/range-rate measurement, and a Joseph-form covariance
  update for symmetry under finite precision.
- `FilteredTrack` wires a seeker to an estimator behind the existing
  `TrackSource` interface. No guidance code changed.
- Measurement noise for the angle channels is range-dependent,
  `angle_sigma^2 + (glint_sigma / range)^2`, so the filter is told the truth
  about the terminal phase: glint's angular effect grows as range falls.
- Phase 5 exit criterion met. On the crossing engagement with a realistic
  seeker, over 6 seeds: median miss falls from **1577 m with no estimator to
  0.88 m** with the EKF, well inside the 5 m lethal radius, and the filter
  coasts through a 0.5 s blackout.
- `examples/estimator_comparison.py` and `plot_estimator_comparison` compare
  five estimator configurations against three target behaviours. The finding:
  heavy alpha-beta smoothing is the best estimator against a straight target
  (0.40 m) and the only outright failure against a 7 g break turn (9.58 m,
  0 hits from 6), while every EKF configuration hits 6 from 6 everywhere
  without ever being the best. A fixed gain must be chosen in advance for
  behaviour that is not knowable in advance.
- Engagements can be defined without editing Python. `interceptor.config` reads
  a scenario from TOML — launch geometry, motor, airframe, the full seeker error
  budget, guidance law, estimator and tuning — and `interceptor.cli` provides
  `list`, `show`, `run` and `sweep` behind an `interceptor` console script. Six
  scenarios ship with the package, including the perfect-information and pure
  pursuit baselines, and `interceptor show <name> --raw` prints one to copy.
- Unrecognised configuration keys are errors, with a suggestion: `seeker.glint`
  reports *did you mean 'glint_sigma'?* rather than silently using the default.
  Every table validates its own key set, types are checked (including that a
  TOML boolean is not accepted as a number, which Python's `bool`/`int`
  relationship would otherwise allow), and ranges are enforced where a negative
  value is meaningless.
- Neither addition brings a dependency. TOML is parsed by `tomllib` and the CLI
  is built on `argparse`, both standard library since 3.11, so the package still
  installs with numpy alone. This deviates from the plan, which named YAML,
  Pydantic and Typer; the reasoning is recorded in `config.py` and `cli.py`, and
  the same argument the project already applied to scipy.
- `interceptor.sensing.consistency` — NEES, the check that asks whether the
  filter's *stated* uncertainty matches its actual error rather than whether the
  estimate is close. `normalised_error_squared` measures the error in units of
  the covariance the filter reports; `chi_squared_interval` gives the acceptance
  bounds by the Wilson-Hilferty transformation, which keeps the package's only
  dependency numpy rather than pulling in scipy for three constants (relative
  error under 3e-4 above 54 degrees of freedom, verified against exact quantiles
  in the tests).
- `tests/test_consistency.py`, including a negative control that aims the same
  machinery at a filter sabotaged to understate its variance a hundredfold and
  requires it to fail. A consistency check that cannot fail certifies nothing.
- Measured result: across 24 independent seeds the EKF scores a mean NEES of
  6.16 against an expected 6.00, inside the 95% interval, and stays inside it
  against a 6 g weave its constant-acceleration model does not describe and
  across a 600-fold range of `jerk_sigma`.

### Changed

- `interceptor.viz.plots` coverage rises from 48% to 97%, and the project from
  86% to 95%. `plot_comparison`, `plot_noise_sweep` and `plot_estimator_comparison`
  had no tests beyond "returns a figure"; they now cover panel structure, both
  themes, the optional baseline and hit-count annotations, unknown-theme
  rejection, the file-writing wrappers, and the degenerate inputs — one law, one
  panel, nothing to compare — where the array-versus-scalar handling of
  `subplots` is easy to get wrong.
- Phase 5's exit criterion was rewritten from "within 2× of the
  perfect-information baseline" (0.06 m) to "inside the 5 m lethal radius". The
  original was unachievable for a physical reason rather than a implementation
  one: glint sets a terminal miss floor that no amount of filtering removes,
  because its angular contribution peaks in the last moments of flight when
  there is no time left to average it away.
- `plot_estimator_comparison` draws dots and ranges rather than bars. Bar length
  is measured from the axis origin, which on a logarithmic axis is arbitrary, so
  bar lengths do not preserve ratios; marker position does. Pass or fail against
  the lethal radius is carried by marker fill rather than a second colour, so it
  survives colour-vision deficiency and monochrome printing.

### Fixed

- **The filter was three-sigma overconfident, and nothing else would have found
  it.** The seeker models one frame of processing latency and stamps each
  measurement with when it was taken. `FilteredTrack` corrected the estimator's
  current-epoch state with that stale measurement, and passed the missile's
  *current* state as the frame to interpret it in — folding one frame of
  relative motion into every estimate as a standing offset. At 650 m/s of
  closing that is 6.5 m, against a filter reporting about 2 m of position
  uncertainty. NEES read 1804 where 6 was expected; with latency disabled it
  read 6.22, which located the fault precisely.

  `FilteredTrack` now keeps the two clocks apart. The estimator is predicted to
  the measurement's own epoch and corrected with the missile state recorded at
  that epoch, and the track handed to the guidance law is extrapolated forward
  from there to the present. The estimator itself needed no change — its
  mathematics was never wrong.

  The correction moves the Phase 5 headline from 0.94 m to 0.88 m, which is the
  point: the bias was nearly invisible in miss distance and would have corrupted
  Phase 7's distribution silently.
- CI's lint job no longer fails on a fresh dependency install. mypy was told to
  target Python 3.11, and numpy's own `.pyi` stubs use PEP 695 `type` statements
  from 2.5 onwards, which mypy refuses to parse under that target — so it failed
  inside numpy before reaching any project code. The type-check target is now
  3.12; genuine 3.11 compatibility is enforced by ruff's `target-version` and by
  the CI matrix running the full suite on a real 3.11 interpreter, which are
  better guarantees than a checker's syntax level anyway. This was latent from
  the moment numpy 2.5 shipped and had nothing to do with Phase 5.
- `plot_comparison` passes its legend corner as a `Literal` rather than a bare
  `str`. matplotlib 3.11 narrowed `loc` in its stubs, and a value reaching
  `legend()` through a loop variable had widened to `str`.

- The plotting module no longer touches `matplotlib.pyplot`. `pyplot.figure()`
  routes through the active backend, which on a desktop machine opens a GUI
  window — pointless for a module that only writes PNG files, and a hard
  failure where Tcl/Tk is missing or incomplete. Figures are now built as bare
  `Figure` objects with an Agg canvas attached, so `interceptor.viz` is
  headless by construction rather than by configuration: no display, no
  `MPLBACKEND`, no figure registry to leak.

### Added

- Phase 4 — the seeker. `GeometricSeeker` computes the true geometry and then
  degrades it: glint applied as a metre-scale wander of the apparent centre (so
  its angular effect grows as range falls), Gaussian noise on range, both
  angles and Doppler, gimbal gating, detection against an `R^-4` signal-to-noise
  law with cross-section scintillation, and a one-frame processing delay.
  `SeekerTrack` reconstructs world-frame relative position from the measured
  range and angles, and — deliberately naively — differences successive
  positions for relative velocity.
- Phase 4 exit criterion met: miss distance rises monotonically from 0.02 m at
  0.02 mrad of angle noise to 1577 m at a realistic 2 mrad, as roughly the
  square of the noise. Proportional navigation survives to about 0.2 mrad.
- `SeekerConfig.perfect()` runs the whole measurement chain with every error
  zeroed. It reproduces the truth-data baseline to 0.013 m, which is what
  proves the frame reconstruction is correct and the degradation above is
  caused by noise rather than by a sign error in the plumbing.

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
