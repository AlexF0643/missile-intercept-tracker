# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-09-06

First release. A 3-DOF engagement simulation with a noisy radar seeker, an
extended Kalman filter, pure pursuit and both forms of proportional navigation,
a 3D viewer, a browser app, and a Monte Carlo that samples the model's own
uncertain constants as well as the seeker's noise. Phases 0 through 7 of the
build plan, complete.

Three results were wrong before they were right, and the corrections are kept
below rather than tidied away: pure pursuit's headline miss was an artefact of a
missile fast enough to overshoot, the filter reported an honest-looking
covariance that was three sigma optimistic, and augmented pronav's failure
against a barrel roll was airframe saturation rather than the broken assumption
it appeared to be.

### Added

- **A Monte Carlo over two kinds of uncertainty, because sampling only one of
  them is misleading.** `interceptor monte-carlo` and
  `examples/monte_carlo.py`. The conventional half flies many seeds at fixed
  constants and reports a probability of kill with a Wilson interval — Wilson
  rather than the normal approximation, which is exactly zero wide at 200 hits
  from 200 and would have the study claiming certainty from a finite sample.
  The other half draws the *constants* from declared ranges and reports the
  spread of probabilities across them.

  On a 6 g weave, 200 launches at the shipped constants:

  | | probability of kill | median miss | worst |
  |---|---|---|---|
  | PN (N=3) | 0.000 [0.000, 0.019] | 6.90 m | 8.93 m |
  | APN (N=3) | 0.970 [0.936, 0.986] | 1.42 m | 30.08 m |

  Across 60 airframes drawn from the ranges in `uncertainty.py`:

  | | probability of kill | |
  |---|---|---|
  | PN (N=3) | 0.00 to 1.00, mean 0.47 | 29 never hit, 23 always |
  | APN (N=3) | 0.00 to 1.00, mean 0.27 | 38 never hit, 9 always |

  **The seeker's noise decides almost nothing.** 87% of PN's draws and 80% of
  APN's are all-or-nothing — every launch hits or none does — which ten seeds
  can only produce if the underlying probability is already pinned near 0 or 1.
  The airframe decides; the noise settles the margin. **And the ordering of the
  two laws reverses**: APN looks decisively better on the shipped constants and
  is worse on average across plausible ones, which puts the previous entry's
  finding in its proper frame.
- **Every constant now says where it came from.** `interceptor/uncertainty.py`
  classifies all 28 as `defined`, `derived`, `design` or `chosen`, with a
  sentence each. The thirteen `chosen` ones — the guesses — carry a range within
  which the truth plausibly lies, and those are what the Monte Carlo samples. No
  entry claims a citation, because none was consulted; where a real programme
  would open a wind-tunnel database this says so and gives a range instead.
  Tested against the browser form, so a setting a person can drag a slider on
  cannot exist without an account of where its default came from.
- A sensitivity pass, reported as a hint rather than a ranking: rank correlation
  between each drawn constant and that draw's probability of kill. Nothing in
  the seeker leads, and top of the list at +0.46 is `max_lift_coefficient` — the
  same constant behind the augmented-pronav misdiagnosis above, found again by a
  method with no connection to that investigation.
- `Study.save`/`load`, because flying is half an hour and drawing is a second.
- **`interceptor serve` — the whole simulation in one browser window.** The 3D
  view on the left, every parameter on the right, re-flown without touching a
  file. `http.server` and a hand-rolled canvas projection: no web framework, no
  three.js, no CDN, so it works with no network at all and the package still
  installs with nothing but numpy. It is a third renderer over the same
  `Storyboard` the GIF recorder and the VPython window consume, so the three
  cannot disagree about where anything is.

  The browser never decides what a valid scenario is. The form builds TOML — the
  same text `interceptor show --raw` prints, shown and editable in a pane — and
  posts it to `config.loads`, so every bound, default and *did you mean
  'glint_sigma'?* is the one the command line already uses. The form itself is
  generated from a table of dotted TOML paths (`web/fields.py`) rather than
  written out, and two tests hold that table to the validator: all sixty
  combinations of manoeuvre, law and estimator must resolve, and every default
  must equal what *omitting* the key would mean. The second caught two form
  defaults that silently disagreed with the file semantics.

  Browser tests run under headless Chromium behind the existing `gui` marker.
  One counts painted pixels, which is the only way to tell a working viewer from
  a canvas that throws inside its render loop while every response stays a
  cheerful 200.
- **Augmented proportional navigation**, `a = N*V_c*(Omega x r_hat) +
  (N/2)*a_t_perp`. The coefficient is the optimal-control solution for a target
  holding *constant* acceleration, under the same criterion that gives `N = 3`
  against one holding none — not a tuning knob. Selected with `law = "apn"`,
  and it needs an estimator that reports acceleration: the EKF does, the
  alpha-beta filter reports zero and APN degrades silently and correctly to
  plain PN behind it. `TruthTrack` now supplies the target's real acceleration
  through a new optional `Entity.commanded_manoeuvre` hook, so APN's ceiling can
  be measured separately from the filter's estimate of it.
- **It recovers both cases induced drag broke, completely.** Through a 2 mrad
  seeker and the EKF, six seeds: the weave goes from 6.94 m and 0 hits from 6 to
  1.21 m and 6 from 6; the break turn from 4.87 m and 3 from 6 to 1.62 m and 6
  from 6. Nothing else changed — same missile, same seeker, same filter.
- **And where it fails, the reason is not the one it looks like.** Against a
  barrel roll APN is twelve times *worse* than the law it augments (469 m
  against 31 m), and it fails that way on a **perfect** track, so the filter is
  not the explanation. Neither is the constant-acceleration premise, even though
  a barrel roll plainly violates it. Hold everything fixed and vary only
  `max_lift_coefficient`, on truth:

  | `Cl_max` | PN | APN |
  |---|---|---|
  | 2.5 | 24.44 m, 18% saturated | 310.98 m, 40% saturated, arrives 265 m/s |
  | 3.5 | 8.61 m, 12% | 7.53 m, 20% |
  | 5.0 | 3.31 m, 7% | **0.02 m**, 9%, arrives 420 m/s |

  Twice the lift and APN is a hundred and fifty times *better* than PN against
  the manoeuvre that supposedly defeats it — the premise is exactly as violated
  at `Cl_max = 5` as at 2.5. **The lead term is a request for lift**, roughly
  half as much again, and it is only free when the airframe has margin. Where it
  saturates the surplus is never produced, but the lift that *is* produced still
  costs induced drag, so the missile pays for the whole command and receives
  part of it. A weave banked 60° out of the horizontal is the same story (306 m
  against PN's 21 m, and 0.41 m once the lift is there): the missile is already
  spending lift on holding itself up. The jink is the one genuine estimator
  failure — APN handles it on truth (1.5 m against 11.1 m) and loses through the
  filter, not because the EKF is slow (measured: it picks up each new break in
  about 20 ms) but because its acceleration error over the last two seconds runs
  at a median 12.0 g against a target pulling 7.0, which APN amplifies by `N/2`
  into the command. Uncomfortably, the number deciding all of this is one of the
  constants this project cannot cite.
- `augmented` ships as a scenario — the break turn APN was derived for, with a
  one-line edit in the comments to switch it to the barrel roll it cannot do on
  the shipped airframe — and `examples/augmented_pronav.py` measures all five
  behaviours on both a perfect track and a real one, writing
  `runs/augmented-pronav.png`.
- **Turning costs energy.** `Aerodynamics` now adds lift-induced drag,
  `Cd = Cd0 + k*Cn^2`. The factor is not a new free constant: a body at
  incidence makes its normal force perpendicular to its own axis rather than to
  the flight path, so the streamwise share is `Cn*sin(alpha) ~ Cn^2 / Cn_alpha`,
  and the lift-curve slope is the peak lift coefficient over the angle it needs
  — both of which the airframe already declared. At full lift that is roughly
  four times the zero-lift drag. Until now the missile manoeuvred for free, and
  every result in the project was flattered by it. The coefficient is an
  engineering estimate rather than a citation, and says so in the source;
  Fleeman or DATCOM would supply a real one.
- Three genuinely three-dimensional target manoeuvres. `barrel_roll` rotates a
  constant-g pull about the flight path, so the target corkscrews — a helix no
  plane contains, and the one manoeuvre that makes the 3D viewer earn its place,
  since it is nearly indistinguishable from a weave in plan view. `jink` pulls
  hard in uncorrelated random directions at a fixed interval, deterministic from
  a seed: where a weave punishes an estimator that smooths too hard, a jink
  punishes any estimator at all. And `weave` and `break_turn` gained `bank_deg`,
  so either can be flown in any plane from a flat turn through a pure pull-up to
  a split-S.
- `barrel-roll` and `jink` ship as scenarios, and the reference file documents
  every new key. Both currently *miss* with a realistic seeker — 31 m and 15 m —
  which is a finding rather than a defect, and the one Phase 7 exists to answer.

### Changed

- **Induced drag overturned a headline result and replaced it with a better
  one.** Pure pursuit used to miss a straight crossing target by 40.8 m; slowed
  by its own turning it no longer overshoots, so it settles into a stern chase
  and now hits by 1 cm. That is not pursuit being rescued: it takes 18.8 s
  against proportional navigation's 13.5, and arrives at 72 m/s of closing speed
  against 284. And the moment the target manoeuvres at all, the old gap returns
  and widens — 89 m against 6 m on a weave, 434 m against 3 m on a break turn.
  The original claim was partly an artefact of a missile fast enough to
  overshoot; the replacement is stronger and rests on time and terminal energy
  rather than miss distance alone.
- Phase 3's exit criterion is now flown against a *weaving* crossing target. A
  criterion that survives a physics correction unchanged was probably measuring
  the wrong thing; this one did not, and the order of magnitude returns against
  a target that does anything at all (14.8×).
- The bearing-drift test measures *spread* rather than peak. Pursuit's
  line-of-sight rate no longer runs away — it sweeps up and then collapses as it
  settles into the chase — so comparing peaks stopped detecting the failure
  while the failure was still there. PN holds the rate inside a factor of 2.1;
  pursuit's varies by four orders of magnitude.
- Every headline number re-measured. Phase 4: 1644 m with no estimator. Phase 5:
  1.05 m with the EKF against a straight target. Against a manoeuvring one the
  filters now **miss** — 6.94 m on the weave, 4.87 m on the break turn, against
  a 5 m lethal radius — because the missile spends its energy turning and
  arrives too slow to correct. Not a regression: the model becoming honest, and
  exactly the gap augmented proportional navigation is meant to close using the
  target acceleration the EKF already estimates.
- **An engagement is 1.85x faster, with the same answers.** A Monte Carlo is a
  few thousand runs, so the two places the profiler pointed at were finally
  worth fixing. The EKF's measurement Jacobian was central differences — 18
  evaluations per cycle, 40% of a flight — and is now the analytic partials of
  range, azimuth, elevation and range-rate. The original argument against
  hand-derived partials still stands (a dropped sign is invisible to review and
  produces a filter that diverges slowly enough to look like a tuning problem),
  so the numerical version stays as the *oracle*: a test compares the two over
  400 random geometries. And `np.cross` and `np.linalg.norm`, called 57,000 and
  273,000 times per engagement, are replaced by three-vector versions written
  out — both tested against the library functions they replace, because
  agreeing with the reference is their entire specification. 3.34 s to 1.80 s,
  and trials then divide across cores through a stdlib process pool.
- `config.bundled_text` returns a shipped scenario's raw TOML, comments and all.
  Both `interceptor show --raw` and the browser app's editor now start there
  rather than reaching into the package's resources themselves.
- `EngagementSpec.describe()` is the one sentence naming the law, the seeker and
  the estimator. There are three things displaying it now, and somebody running
  the same scenario two ways should not have to work out whether two differently
  worded descriptions mean the same configuration.
- `Frame` carries the index of the recorded sample it was drawn from. The
  renderers that hold one frame at a time keep using the cumulative trails; the
  browser, which would otherwise send a growing trail per frame down a wire,
  indexes into the two trajectories instead.
- `interceptor show` and the 3D viewer's subtitle name the guidance law by its
  own name rather than its class's — `ProNav (N=3)` against `APN (N=3)`, where
  the class names differ only by a prefix and the navigation constant, which is
  what actually changes the behaviour, appeared in neither.
- `examples/compare_laws.py` reports time to intercept and closing speed
  alongside miss distance, and adds a weaving case. Miss distance alone cannot
  tell a clean intercept from a nineteen-second stern chase.
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
- CI's lint job no longer breaks when the toolchain moves under it. Two
  separate cases, both from tools newer than the ones on the development
  machine. A `type: ignore` on the `FuncAnimation` callback became *unused*
  once matplotlib loosened that annotation, and under `strict` an unnecessary
  ignore is itself an error — so the suppression failed on exactly the upgrade
  it existed to survive. It is replaced by a small adapter returning an empty
  artist list, which satisfies every version of the stub without suppressing
  anything. And ruff 0.16 began formatting Python inside Markdown, which
  reformatted the README and failed `ruff format --check` on a commit that had
  not touched a line of Python; ruff is now pinned to a minor range, because a
  formatter is not a dependency worth floating.
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

- **A comparison figure whose panels could come out blank.**
  `plot_estimator_comparison` built its panels with `sharex` and then let each
  panel set its own x limits, so the last one silently won and any panel whose
  data was smaller fell off the left edge — an axis with nothing in view draws
  perfectly happily and warns about nothing. It went unnoticed while every panel
  held similar numbers; comparing PN with APN, where one panel runs to 469 m and
  another sits at 1 m, left three of five panels empty. The limits are now
  computed once over every panel, which is what sharing a scale was supposed to
  mean. Header spacing is stated in inches rather than figure fractions for the
  same class of reason: at two rows instead of five the caption landed on top of
  the title.
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
[0.1.0]: https://github.com/AlexF0643/missile-intercept-tracker/releases/tag/v0.1.0
