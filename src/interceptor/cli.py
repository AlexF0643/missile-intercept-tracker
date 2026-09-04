"""Run engagements from the command line.

    interceptor list
    interceptor show crossing
    interceptor run crossing --seed 3
    interceptor run my-scenario.toml --figure out.png
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
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from interceptor.config import ConfigError, EngagementSpec, bundled_names, load, load_bundled
from interceptor.sim.engagement import run

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


def _fly(spec: EngagementSpec, seed: int) -> tuple[object, object]:
    world, detector = spec.build(seed=seed)
    result = run(world, duration=spec.scenario.duration, dt=PHYSICS_STEP, stop=detector)
    return result, detector.result


def _describe(spec: EngagementSpec) -> str:
    law = "unguided" if spec.law is None else type(spec.law).__name__
    if spec.seeker is None:
        # An estimator behind a perfect track has nothing to estimate, and
        # `Scenario.build` ignores it. Saying which one is configured would
        # imply it was running.
        return f"{law}, perfect information"
    seeker = f"seeker {spec.seeker.angle_sigma * 1e3:g} mrad"
    estimator = "no estimator" if spec.estimator is None else spec.estimator().name
    return f"{law}, {seeker}, {estimator}"


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
        from importlib import resources

        from interceptor.config import BUNDLED

        if args.scenario in bundled_names():
            print(
                resources.files(BUNDLED)
                .joinpath(f"{args.scenario}.toml")
                .read_text(encoding="utf-8"),
                end="",
            )
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

    miss = intercept.miss_distance  # type: ignore[attr-defined]
    hit = intercept.hit  # type: ignore[attr-defined]
    print(f"  miss distance   {miss:>10.3f} m   {'HIT' if hit else 'miss'}")
    print(f"  at              {intercept.time:>10.3f} s")  # type: ignore[attr-defined]

    if args.figure:
        try:
            from interceptor.viz.plots import save_engagement
        except ImportError:
            print("\ncannot draw a figure without matplotlib: pip install -e '.[viz]'")
            return 1
        path = save_engagement(
            result,  # type: ignore[arg-type]
            args.figure,
            intercept=intercept,  # type: ignore[arg-type]
            title=f"{spec.name} — {_describe(spec)}",
        )
        print(f"  figure          {path}")
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
        misses.append(intercept.miss_distance)  # type: ignore[attr-defined]
        hits += bool(intercept.hit)  # type: ignore[attr-defined]

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

    sweep = commands.add_parser("sweep", help="fly one engagement under many seeds")
    sweep.add_argument("scenario", help="bundled name or path to a .toml file")
    sweep.add_argument("--seeds", type=int, default=10, help="how many runs (default: 10)")
    sweep.set_defaults(handler=_command_sweep)

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
