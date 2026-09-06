"""Run engagements from the command line.

    interceptor list
    interceptor show crossing
    interceptor run crossing --seed 3
    interceptor run my-scenario.toml --figure out.png
    interceptor record crossing -o flight.gif
    interceptor view crossing
    interceptor sweep crossing --seeds 20

**Why argparse and not Typer.** The plan said Typer, and Typer is pleasant. It
is also a dependency, and so are the two it brings with it, for a program with
four subcommands and about a dozen options. ``argparse`` is in the standard
library and does this much perfectly well, which keeps the package installable
with nothing but numpy. The same argument as for TOML over YAML in
:mod:`interceptor.config`, and for Wilson-Hilferty over scipy in
:mod:`interceptor.sensing.consistency`: a dependency should buy something the
standard library cannot.

Matplotlib stays optional and is imported only when a figure is asked for, so
``interceptor run`` works on a machine that has never heard of it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from interceptor.config import (
    ConfigError,
    EngagementSpec,
    bundled_names,
    bundled_text,
    load,
    load_bundled,
)
from interceptor.sim.engagement import RunResult, run
from interceptor.sim.intercept import Intercept

if TYPE_CHECKING:
    from interceptor.viz.scene import Storyboard

__all__ = ["main"]

PHYSICS_STEP = 1e-3


def _resolve(reference: str) -> EngagementSpec:
    """A bundled name or a path to a file, whichever the argument looks like.

    Checked in that order deliberately: a bundled name is what someone types
    from memory, and a path is unambiguous because it ends in ``.toml``.
    """
    if reference.endswith(".toml") or Path(reference).exists():
        return load(reference)
    return load_bundled(reference)


def _fly(spec: EngagementSpec, seed: int) -> tuple[RunResult, Intercept | None]:
    world, detector = spec.build(seed=seed)
    result = run(world, duration=spec.scenario.duration, dt=PHYSICS_STEP, stop=detector)
    return result, detector.result


def _describe(spec: EngagementSpec) -> str:
    # One sentence, defined on the spec, so the command line, the 3D viewer and
    # the browser app cannot describe the same configuration differently.
    return spec.describe()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def _command_list(args: argparse.Namespace) -> int:
    names = bundled_names()
    width = max(len(name) for name in names)
    for name in names:
        print(f"  {name:<{width}}  {load_bundled(name).description}")
    print(f"\nRun one with:  interceptor run {names[0]}")
    print("Copy one to edit:  interceptor show " + names[0] + " --raw > my-scenario.toml")
    return 0


def _command_show(args: argparse.Namespace) -> int:
    if args.raw:
        if args.scenario in bundled_names():
            print(bundled_text(args.scenario), end="")
            return 0
        print(Path(args.scenario).read_text(encoding="utf-8"), end="")
        return 0

    spec = _resolve(args.scenario)
    scenario = spec.scenario
    separation = float(np.linalg.norm(scenario.target_position - scenario.missile_position))
    print(f"{spec.name}: {spec.description}")
    print(f"  {_describe(spec)}")
    print(f"  initial separation   {separation:>10.0f} m")
    print(f"  target speed         {float(np.linalg.norm(scenario.target_velocity)):>10.0f} m/s")
    print(f"  missile mass         {scenario.missile_mass:>10.1f} kg")
    print(f"  airframe limit       {scenario.aero.max_lateral_g:>10.1f} g")
    print(f"  lethal radius        {scenario.lethal_radius:>10.1f} m")
    print(f"  duration             {scenario.duration:>10.1f} s")
    return 0


def _command_run(args: argparse.Namespace) -> int:
    spec = _resolve(args.scenario)
    print(f"{spec.name}: {_describe(spec)}")

    result, intercept = _fly(spec, args.seed)
    if intercept is None:
        print("no closest approach within the scenario duration — the missile never got near")
        return 1

    miss = intercept.miss_distance
    hit = intercept.hit
    print(f"  miss distance   {miss:>10.3f} m   {'HIT' if hit else 'miss'}")
    print(f"  at              {intercept.time:>10.3f} s")

    if args.figure:
        try:
            from interceptor.viz.plots import save_engagement
        except ImportError:
            print("\ncannot draw a figure without matplotlib: pip install -e '.[viz]'")
            return 1
        path = save_engagement(
            result,
            args.figure,
            intercept=intercept,
            title=f"{spec.name} — {_describe(spec)}",
        )
        print(f"  figure          {path}")
    return 0


def _storyboard(
    spec: EngagementSpec, args: argparse.Namespace
) -> tuple[Storyboard, Intercept | None]:
    """Fly the engagement and turn it into frames."""
    from interceptor.viz.scene import storyboard_from

    result, intercept = _fly(spec, args.seed)
    slow_from: float | None = None
    if args.slow_motion and intercept is not None:
        # The endgame is what anybody watches for and it is over in moments.
        slow_from = max(intercept.time - 1.5, 0.0)
    # A shorter title than `run` prints: this one has to fit inside a 578-pixel
    # frame, and the estimator's tuning is not what a viewer is watching for.
    estimator = "" if spec.estimator is None else type(spec.estimator()).__name__
    law = "unguided" if spec.law is None else spec.law.name
    if spec.seeker is None:
        subtitle = f"{law}, perfect information"
    else:
        subtitle = f"{law}, {spec.seeker.angle_sigma * 1e3:g} mrad seeker, {estimator}"

    return storyboard_from(
        result,
        fps=args.fps,
        intercept=intercept,
        title=f"{spec.name} — {subtitle}",
        slow_motion_from=slow_from,
    ), intercept


def _command_record(args: argparse.Namespace) -> int:
    spec = _resolve(args.scenario)
    print(f"{spec.name}: {_describe(spec)}")
    try:
        from interceptor.viz.record import save_flight
    except ImportError:
        print("recording needs matplotlib: pip install -e '.[viz]'")
        return 1

    storyboard, intercept = _storyboard(spec, args)
    if intercept is not None:
        verdict = "HIT" if intercept.hit else "miss"
        print(f"  miss distance   {intercept.miss_distance:>10.3f} m   {verdict}")
    print(f"  frames          {len(storyboard.frames):>10}")

    path = save_flight(storyboard, args.output, orbit=args.orbit)
    print(f"  written         {path}")
    return 0


def _command_view(args: argparse.Namespace) -> int:
    spec = _resolve(args.scenario)
    print(f"{spec.name}: {_describe(spec)}")
    try:
        from interceptor.viz.live import view
    except ImportError:
        print(
            "the live viewer needs the scene model, which needs numpy only — "
            "this should not happen; please report it"
        )
        return 1

    storyboard, _ = _storyboard(spec, args)
    print(f"  {len(storyboard.frames)} frames; opening a browser window")
    try:
        view(storyboard, loop=args.loop)
    except RuntimeError as error:
        print(f"\n{error}")
        return 1
    return 0


def _command_monte_carlo(args: argparse.Namespace) -> int:
    """Fly a scenario many times, over noise and over the constants.

    `sweep` varies the seeker's noise, which is the uncertainty everyone
    remembers. This also varies the numbers nobody looked up, which is usually
    the larger of the two and is the whole reason the command exists.
    """
    import tomllib

    from interceptor.sim.montecarlo import study, wilson_interval

    text = (
        bundled_text(args.scenario)
        if args.scenario in bundled_names()
        else Path(args.scenario).read_text(encoding="utf-8")
    )
    scenario = tomllib.loads(text)
    spec = _resolve(args.scenario)

    total = args.draws * args.seeds
    print(f"{spec.name}: {_describe(spec)}")
    print(f"  {args.draws} parameter draws x {args.seeds} seeds = {total} engagements")
    if args.draws == 1:
        print("  (--draws 1 is noise only: the conventional answer, and the misleading one)")

    started = time.perf_counter()
    result = study(
        scenario,
        draws=args.draws,
        seeds=args.seeds,
        workers=args.workers,
        name=spec.name,
        rng_seed=args.rng_seed,
    )
    elapsed = time.perf_counter() - started

    nominal = result.nominal_trials()
    hits = sum(t.hit for t in nominal)
    low, high = wilson_interval(hits, len(nominal))
    misses = np.array([t.miss_distance for t in nominal])

    print(f"\n  flown in {elapsed:.0f} s")
    print("\n  With the constants this project ships:")
    print(f"    probability of kill   {hits / len(nominal):>8.2f}  [{low:.2f}, {high:.2f}] 95%")
    print(f"    median miss           {np.median(misses):>8.2f} m")
    print(f"    90th percentile       {np.percentile(misses, 90):>8.2f} m")
    print(f"    worst                 {misses.max():>8.2f} m")

    if args.draws > 1:
        probabilities = np.array(result.kill_probabilities())
        print("\n  Across constants that are equally plausible:")
        print(f"    probability of kill   {probabilities.min():.2f} to {probabilities.max():.2f}")
        print(f"    median across draws   {np.median(probabilities):>8.2f}")
        print(
            "\n  The second range is the one to quote. The first assumes the guesses"
            "\n  in interceptor/uncertainty.py are right."
        )
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    """Open the whole thing in a browser window.

    The one command that does not take a scenario: the point of it is that you
    choose and change the scenario in the window, without coming back here.
    """
    from interceptor.web.app import serve

    serve(
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        verbose=args.verbose,
    )
    return 0


def _command_sweep(args: argparse.Namespace) -> int:
    """Fly the same engagement under many seeds.

    One run of a stochastic engagement is an anecdote. The seeker's noise is
    seeded, so a single miss distance says as much about the draw as about the
    configuration, which is why every result in this project is a median over
    seeds rather than a number from one flight.
    """
    spec = _resolve(args.scenario)
    print(f"{spec.name}: {_describe(spec)}, {args.seeds} seeds")

    misses, hits = [], 0
    for seed in range(args.seeds):
        _, intercept = _fly(spec, seed)
        if intercept is None:
            continue
        misses.append(intercept.miss_distance)
        hits += bool(intercept.hit)

    if not misses:
        print("no run produced a closest approach")
        return 1

    row = np.array(misses)
    print(f"  median          {np.median(row):>10.3f} m")
    print(f"  best            {row.min():>10.3f} m")
    print(f"  worst           {row.max():>10.3f} m")
    print(f"  hits            {hits:>10} / {args.seeds}")
    return 0


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interceptor",
        description="Fly missile intercept engagements defined in TOML files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "A scenario is either a bundled name (see 'interceptor list') or a path\n"
            "to a .toml file. Start from a bundled one:\n\n"
            "    interceptor show crossing --raw > my-scenario.toml\n"
            "    interceptor run my-scenario.toml\n"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="show the scenarios that ship with the package")
    listing.set_defaults(handler=_command_list)

    show = commands.add_parser("show", help="describe a scenario without flying it")
    show.add_argument("scenario", help="bundled name or path to a .toml file")
    show.add_argument("--raw", action="store_true", help="print the file itself, for copying")
    show.set_defaults(handler=_command_show)

    fly = commands.add_parser("run", help="fly one engagement")
    fly.add_argument("scenario", help="bundled name or path to a .toml file")
    fly.add_argument("--seed", type=int, default=0, help="seeker noise seed (default: 0)")
    fly.add_argument("--figure", type=Path, help="write a five-panel diagnostic figure here")
    fly.set_defaults(handler=_command_run)

    record = commands.add_parser("record", help="write an animation of one engagement")
    record.add_argument("scenario", help="bundled name or path to a .toml file")
    record.add_argument("--seed", type=int, default=0, help="seeker noise seed (default: 0)")
    record.add_argument(
        "-o", "--output", type=Path, default=Path("flight.gif"), help="output .gif or .mp4"
    )
    record.add_argument("--fps", type=float, default=20.0, help="frames per second (default: 20)")
    record.add_argument(
        "--orbit",
        action="store_true",
        help="drift the camera for parallax; roughly triples the file size",
    )
    record.add_argument(
        "--no-slow-motion",
        dest="slow_motion",
        action="store_false",
        help="play the endgame at full speed rather than quarter speed",
    )
    record.set_defaults(handler=_command_record)

    watch = commands.add_parser("view", help="play one engagement in an interactive 3D window")
    watch.add_argument("scenario", help="bundled name or path to a .toml file")
    watch.add_argument("--seed", type=int, default=0, help="seeker noise seed (default: 0)")
    watch.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30)")
    watch.add_argument("--loop", action="store_true", help="replay until the window is closed")
    watch.add_argument(
        "--no-slow-motion", dest="slow_motion", action="store_false", help="no endgame slow motion"
    )
    watch.set_defaults(handler=_command_view)

    app = commands.add_parser(
        "serve", help="open the whole simulation in a browser window: 3D view and every parameter"
    )
    app.add_argument("--port", type=int, default=8765, help="port to listen on (default: 8765)")
    app.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default: 127.0.0.1, i.e. this machine only)",
    )
    app.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="do not open a browser; just print the address",
    )
    app.add_argument("--verbose", action="store_true", help="log every request")
    app.set_defaults(handler=_command_serve)

    sweep = commands.add_parser("sweep", help="fly one engagement under many seeds")
    sweep.add_argument("scenario", help="bundled name or path to a .toml file")
    sweep.add_argument("--seeds", type=int, default=10, help="how many runs (default: 10)")
    sweep.set_defaults(handler=_command_sweep)

    carlo = commands.add_parser(
        "monte-carlo",
        help="fly a scenario many times over noise AND over the uncited constants",
    )
    carlo.add_argument("scenario", help="bundled name or path to a .toml file")
    carlo.add_argument(
        "--draws",
        type=int,
        default=20,
        help="sets of constants to try, drawn from the ranges in "
        "interceptor/uncertainty.py (default: 20; 1 means noise only)",
    )
    carlo.add_argument("--seeds", type=int, default=50, help="noise seeds per draw (default: 50)")
    carlo.add_argument(
        "--workers", type=int, default=None, help="processes (default: one per core)"
    )
    carlo.add_argument(
        "--rng-seed", type=int, default=20260906, help="fixes which constants are drawn"
    )
    carlo.set_defaults(handler=_command_monte_carlo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status rather than raising.

    A bad scenario file is a user error, not a crash, so it prints one line and
    exits non-zero instead of presenting a traceback that begins somewhere in
    ``tomllib``.
    """
    args = _parser().parse_args(argv)
    try:
        exit_code: int = args.handler(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
