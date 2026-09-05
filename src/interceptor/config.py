"""Engagements defined in a file rather than in Python.

Until now, changing a target's speed meant editing a source file. That is fine
while the only person running the simulation is the person writing it, and
stops being fine the moment anyone wants to ask a question the code does not
already answer.

**Why TOML and not YAML.** The plan said YAML; this is TOML, and the reason is
that ``tomllib`` has been in the standard library since Python 3.11 while YAML
is a dependency. Scenario files are flat tables of numbers, which is precisely
what TOML is good at, and it is the format the project's own ``pyproject.toml``
already uses, so there is one syntax to learn rather than two. YAML's extra
expressiveness — anchors, multiple documents, implicit typing — buys nothing
here and costs the package a runtime dependency it otherwise does not have.

**Why the validation is hand-written.** The plan said Pydantic. Pydantic is
excellent and would work, but it is a large compiled dependency for roughly
thirty fields, and the one thing it is really needed for — telling a person
exactly which line of their file is wrong — is a couple of hundred lines here.
The package's only runtime requirement stays numpy, which means the simulation
core installs anywhere, and that is worth more than the convenience. Swapping
in Pydantic later would be a contained change: replace this module's readers,
keep :class:`EngagementSpec`.

**Unknown keys are errors.** A configuration format that silently ignores
``anglesigma = 0.002`` will eventually cost someone an afternoon wondering why
the noise setting does nothing. Every table here rejects keys it does not
recognise, and the error names the offending key and suggests the nearest
legitimate one.
"""

from __future__ import annotations

import difflib
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.propulsion import Motor
from interceptor.core.state import Vector
from interceptor.entities.target import (
    Manoeuvre,
    barrel_roll,
    break_turn,
    jink,
    straight_and_level,
    weave,
)
from interceptor.guidance.base import GuidanceLaw
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit
from interceptor.sensing.filters import AlphaBeta, Estimator, ExtendedKalman
from interceptor.sensing.seeker import SeekerConfig
from interceptor.sim.scenarios import Scenario

__all__ = [
    "ConfigError",
    "EngagementSpec",
    "bundled_names",
    "load",
    "load_bundled",
    "loads",
    "resolve",
]

BUNDLED = "interceptor.presets"
"""Where the shipped scenario files live. Named presets rather than scenarios so
that it is never mistaken for ``interceptor.sim.scenarios``, which holds the
Python-defined geometries the tests are built on."""


class ConfigError(ValueError):
    """A scenario file that cannot be turned into an engagement.

    Carries the dotted path of the offending key, because "invalid value" three
    levels into a nested table is not a useful thing to tell anybody.
    """


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------
def _fail(path: str, problem: str) -> ConfigError:
    return ConfigError(f"{path or 'scenario'}: {problem}")


def _known(table: dict[str, Any], allowed: Iterable[str], path: str) -> None:
    """Reject keys that are not part of the schema.

    The suggestion matters more than the rejection. Someone who wrote
    ``glint`` for ``glint_sigma`` wants to be told the name, not merely that
    they were wrong.
    """
    permitted = set(allowed)
    for key in table:
        if key in permitted:
            continue
        close = difflib.get_close_matches(key, sorted(permitted), n=1, cutoff=0.6)
        hint = (
            f" — did you mean {close[0]!r}?" if close else f" (expected one of {sorted(permitted)})"
        )
        raise _fail(f"{path}.{key}" if path else key, f"unknown key{hint}")


def _table(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise _fail(
            f"{path}.{key}" if path else key, f"expected a table, got {type(value).__name__}"
        )
    return value


def _number(
    table: dict[str, Any],
    key: str,
    default: float,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = table.get(key, default)
    where = f"{path}.{key}" if path else key
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(where, f"expected a number, got {type(value).__name__}")
    number = float(value)
    if minimum is not None and number < minimum:
        raise _fail(where, f"must be at least {minimum}, got {number:g}")
    if maximum is not None and number > maximum:
        raise _fail(where, f"must be at most {maximum}, got {number:g}")
    return number


def _integer(table: dict[str, Any], key: str, default: int, path: str, *, minimum: int = 0) -> int:
    value = table.get(key, default)
    where = f"{path}.{key}" if path else key
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(where, f"expected a whole number, got {type(value).__name__}")
    if value < minimum:
        raise _fail(where, f"must be at least {minimum}, got {value}")
    return value


def _flag(table: dict[str, Any], key: str, default: bool, path: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        where = f"{path}.{key}" if path else key
        raise _fail(where, f"expected true or false, got {type(value).__name__}")
    return value


def _text(table: dict[str, Any], key: str, default: str, path: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        where = f"{path}.{key}" if path else key
        raise _fail(where, f"expected a string, got {type(value).__name__}")
    return value


def _choice(
    table: dict[str, Any], key: str, options: Iterable[str], default: str, path: str
) -> str:
    value = _text(table, key, default, path)
    permitted = list(options)
    if value not in permitted:
        raise _fail(f"{path}.{key}" if path else key, f"must be one of {permitted}, got {value!r}")
    return value


def _vector(table: dict[str, Any], key: str, path: str, default: Vector | None = None) -> Vector:
    where = f"{path}.{key}" if path else key
    if key not in table:
        if default is None:
            raise _fail(where, "is required")
        return default
    value = table[key]
    if not isinstance(value, list) or len(value) != 3:
        raise _fail(where, "expected three numbers, [east, north, up]")
    for component in value:
        if isinstance(component, bool) or not isinstance(component, int | float):
            raise _fail(where, f"expected numbers, found {type(component).__name__}")
    return np.array([float(c) for c in value], dtype=np.float64)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def _read_manoeuvre(table: dict[str, Any], path: str) -> Manoeuvre:
    kind = _choice(
        table,
        "kind",
        ("straight", "weave", "break_turn", "barrel_roll", "jink"),
        "straight",
        path,
    )
    if kind == "straight":
        _known(table, {"kind"}, path)
        return straight_and_level()

    if kind == "weave":
        _known(table, {"kind", "amplitude_g", "period", "bank_deg"}, path)
        return weave(
            amplitude_g=_number(table, "amplitude_g", 6.0, path, minimum=0.0),
            period=_number(table, "period", 4.0, path, minimum=1e-3),
            bank_deg=_number(table, "bank_deg", 0.0, path, minimum=-180.0, maximum=180.0),
        )

    if kind == "barrel_roll":
        _known(table, {"kind", "amplitude_g", "period"}, path)
        return barrel_roll(
            amplitude_g=_number(table, "amplitude_g", 5.0, path, minimum=0.0),
            period=_number(table, "period", 4.0, path, minimum=1e-3),
        )

    if kind == "jink":
        _known(table, {"kind", "amplitude_g", "interval", "seed"}, path)
        return jink(
            amplitude_g=_number(table, "amplitude_g", 7.0, path, minimum=0.0),
            interval=_number(table, "interval", 1.5, path, minimum=1e-3),
            seed=_integer(table, "seed", 0, path),
        )

    _known(table, {"kind", "amplitude_g", "start_time", "bank_deg"}, path)
    return break_turn(
        amplitude_g=_number(table, "amplitude_g", 7.0, path, minimum=0.0),
        start_time=_number(table, "start_time", 0.0, path, minimum=0.0),
        bank_deg=_number(table, "bank_deg", 0.0, path, minimum=-180.0, maximum=180.0),
    )


def _read_motor(table: dict[str, Any], path: str) -> Motor:
    _known(
        table,
        {
            "boost_thrust",
            "boost_duration",
            "sustain_thrust",
            "sustain_duration",
            "specific_impulse",
        },
        path,
    )
    return Motor(
        boost_thrust=_number(table, "boost_thrust", 20_000.0, path, minimum=0.0),
        boost_duration=_number(table, "boost_duration", 2.5, path, minimum=0.0),
        sustain_thrust=_number(table, "sustain_thrust", 0.0, path, minimum=0.0),
        sustain_duration=_number(table, "sustain_duration", 0.0, path, minimum=0.0),
        specific_impulse=_number(table, "specific_impulse", 240.0, path, minimum=1.0),
    )


def _read_aero(table: dict[str, Any], path: str) -> Aerodynamics:
    _known(
        table,
        {
            "drag_coefficient",
            "reference_area",
            "max_lateral_g",
            "max_lift_coefficient",
            "peak_lift_angle_deg",
        },
        path,
    )
    return Aerodynamics(
        drag_coefficient=_number(table, "drag_coefficient", 0.30, path, minimum=0.0),
        reference_area=_number(table, "reference_area", 0.02, path, minimum=1e-6),
        max_lateral_g=_number(table, "max_lateral_g", 30.0, path, minimum=0.0),
        max_lift_coefficient=_number(table, "max_lift_coefficient", 2.50, path, minimum=1e-3),
        peak_lift_angle_deg=_number(
            table, "peak_lift_angle_deg", 25.0, path, minimum=1.0, maximum=89.0
        ),
    )


def _read_guidance(table: dict[str, Any], path: str) -> GuidanceLaw | None:
    law = _choice(table, "law", ("pronav", "pursuit", "none"), "pronav", path)
    if law == "none":
        _known(table, {"law"}, path)
        return None
    if law == "pursuit":
        _known(table, {"law"}, path)
        return PurePursuit()
    _known(table, {"law", "navigation_constant"}, path)
    return ProportionalNavigation(_number(table, "navigation_constant", 3.0, path, minimum=0.0))


def _read_seeker(table: dict[str, Any], path: str) -> SeekerConfig | None:
    """``None`` means perfect information, which is the Phase 3 baseline."""
    if not _flag(table, "enabled", True, path):
        return None
    _known(
        table,
        {
            "enabled",
            "range_sigma",
            "angle_sigma",
            "range_rate_sigma",
            "glint_sigma",
            "gimbal_limit_deg",
            "detection_range",
            "snr_fluctuation_db",
            "latency_frames",
            "detection_threshold_db",
        },
        path,
    )
    return SeekerConfig(
        range_sigma=_number(table, "range_sigma", 3.0, path, minimum=0.0),
        angle_sigma=_number(table, "angle_sigma", 2.0e-3, path, minimum=0.0),
        range_rate_sigma=_number(table, "range_rate_sigma", 0.4, path, minimum=0.0),
        glint_sigma=_number(table, "glint_sigma", 1.5, path, minimum=0.0),
        # Degrees in the file, radians in the code. A gimbal limit is something
        # a person states in degrees, and making them write 0.698 would invite
        # exactly one kind of mistake.
        gimbal_limit=np.deg2rad(
            _number(table, "gimbal_limit_deg", 40.0, path, minimum=0.0, maximum=180.0)
        ),
        detection_range=_number(table, "detection_range", 12_000.0, path, minimum=0.0),
        snr_fluctuation_db=_number(table, "snr_fluctuation_db", 3.0, path, minimum=0.0),
        latency_frames=_integer(table, "latency_frames", 1, path),
        detection_threshold_db=_number(table, "detection_threshold_db", 12.0, path),
    )


def _read_estimator(table: dict[str, Any], path: str) -> Callable[[], Estimator] | None:
    """Returns a *factory*, because a filter carries state.

    Reusing one instance across runs would let the second run begin already
    convinced of where a different target was.
    """
    kind = _choice(table, "kind", ("none", "alpha_beta", "ekf"), "ekf", path)
    if kind == "none":
        _known(table, {"kind"}, path)
        return None

    if kind == "alpha_beta":
        _known(table, {"kind", "alpha", "beta"}, path)
        alpha = _number(table, "alpha", 0.25, path, minimum=1e-6, maximum=1.0 - 1e-6)
        beta = (
            _number(table, "beta", 0.0, path, minimum=1e-6, maximum=2.0 - 1e-6)
            if "beta" in table
            else None
        )
        return lambda: AlphaBeta(alpha=alpha, beta=beta)

    _known(
        table,
        {"kind", "jerk_sigma", "initial_velocity_sigma", "initial_acceleration_sigma"},
        path,
    )
    jerk = _number(table, "jerk_sigma", 60.0, path, minimum=0.0)
    velocity_sigma = _number(table, "initial_velocity_sigma", 400.0, path, minimum=1e-6)
    acceleration_sigma = _number(table, "initial_acceleration_sigma", 100.0, path, minimum=1e-6)
    return lambda: ExtendedKalman(
        jerk_sigma=jerk,
        initial_velocity_sigma=velocity_sigma,
        initial_acceleration_sigma=acceleration_sigma,
    )


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EngagementSpec:
    """Everything needed to fly one engagement, read from a file.

    Deliberately the same four arguments :meth:`Scenario.build` already takes,
    so this module is a *reader* and not a second way to describe an engagement.
    Adding a physics option means adding it to ``Scenario``; this file only
    learns how to spell it.
    """

    name: str
    description: str
    scenario: Scenario
    law: GuidanceLaw | None
    seeker: SeekerConfig | None
    estimator: Callable[[], Estimator] | None

    def build(self, seed: int = 0) -> Any:
        """Construct the world and closest-approach detector for one run."""
        return self.scenario.build(
            self.law,
            seeker=self.seeker,
            estimator=None if self.estimator is None else self.estimator(),
            seed=seed,
        )


def resolve(data: dict[str, Any], name: str = "scenario") -> EngagementSpec:
    """Turn already-parsed TOML into a specification."""
    _known(
        data,
        {
            "name",
            "description",
            "duration",
            "lethal_radius",
            "missile",
            "target",
            "guidance",
            "seeker",
            "estimator",
        },
        "",
    )

    missile = _table(data, "missile", "")
    _known(
        missile,
        {"position", "velocity", "speed", "mass", "autopilot_lag", "motor", "aero"},
        "missile",
    )
    target = _table(data, "target", "")
    _known(target, {"position", "velocity", "manoeuvre"}, "target")

    missile_position = _vector(missile, "position", "missile")
    target_position = _vector(target, "position", "target")

    if "velocity" in missile:
        missile_velocity = _vector(missile, "velocity", "missile")
    else:
        # Aimed at the target at a stated speed — how a launch is actually
        # described, and it keeps the two positions as the single source of
        # truth for the geometry.
        speed = _number(missile, "speed", 60.0, "missile", minimum=0.0)
        direction = target_position - missile_position
        span = float(np.linalg.norm(direction))
        if span < 1e-9:
            raise _fail("target.position", "coincides with the missile; there is nothing to aim at")
        missile_velocity = np.asarray(speed * direction / span, dtype=np.float64)

    scenario = Scenario(
        name=_text(data, "name", name, ""),
        missile_position=missile_position,
        missile_velocity=missile_velocity,
        target_position=target_position,
        target_velocity=_vector(target, "velocity", "target"),
        manoeuvre=_read_manoeuvre(_table(target, "manoeuvre", "target"), "target.manoeuvre"),
        missile_mass=_number(missile, "mass", 85.0, "missile", minimum=1e-3),
        motor=_read_motor(_table(missile, "motor", "missile"), "missile.motor"),
        aero=_read_aero(_table(missile, "aero", "missile"), "missile.aero"),
        autopilot_lag=_number(missile, "autopilot_lag", 0.20, "missile", minimum=0.0),
        lethal_radius=_number(data, "lethal_radius", 5.0, "", minimum=0.0),
        duration=_number(data, "duration", 30.0, "", minimum=1e-3),
    )

    return EngagementSpec(
        name=scenario.name,
        description=_text(data, "description", "", ""),
        scenario=scenario,
        law=_read_guidance(_table(data, "guidance", ""), "guidance"),
        seeker=_read_seeker(_table(data, "seeker", ""), "seeker"),
        estimator=_read_estimator(_table(data, "estimator", ""), "estimator"),
    )


def loads(text: str, name: str = "scenario") -> EngagementSpec:
    """Parse a scenario from TOML text."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{name}: not valid TOML — {error}") from error
    return resolve(data, name)


def load(path: str | Path) -> EngagementSpec:
    """Read a scenario from a ``.toml`` file."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read {file}: {error}") from error
    return loads(text, file.stem)


def bundled_names() -> list[str]:
    """The scenarios that ship with the package, alphabetically."""
    return sorted(
        entry.name.removesuffix(".toml")
        for entry in resources.files(BUNDLED).iterdir()
        if entry.name.endswith(".toml")
    )


def load_bundled(name: str) -> EngagementSpec:
    """Read one of the scenarios that ships with the package."""
    available = bundled_names()
    if name not in available:
        close = difflib.get_close_matches(name, available, n=1, cutoff=0.5)
        hint = f" — did you mean {close[0]!r}?" if close else f" (available: {available})"
        raise ConfigError(f"no bundled scenario named {name!r}{hint}")
    text = resources.files(BUNDLED).joinpath(f"{name}.toml").read_text(encoding="utf-8")
    return loads(text, name)
