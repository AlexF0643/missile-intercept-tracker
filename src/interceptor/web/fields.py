"""What the browser form offers, and where each control writes.

The form is generated from this table rather than written out in HTML, for the
same reason the scenario files are validated in one place: two hand-maintained
lists of the same thing drift, and the drift is silent. A control here names the
dotted TOML path it writes, so the page builds a scenario file and the server
reads it with :func:`interceptor.config.loads` — the identical code path a file
on disk goes through, bounds, defaults, suggestions and all.

The bounds below are *not* the validator's. They are what a slider should let
you reach: the validator says a drag coefficient must be non-negative, which is
true and useless for choosing a range to drag across. Anything outside these
bounds is still reachable by typing it into the TOML pane, and will be accepted
if the validator accepts it. The two are allowed to disagree; only one of them
decides.

``when`` is what keeps the form honest about conditional keys. The validator
rejects ``period`` under a break turn rather than ignoring it, so a form that
emitted every key regardless would produce scenarios that cannot be read. A
field appears — and is written — only when its controlling field has one of the
listed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["FIELDS", "GROUPS", "as_json", "default_scenario"]


@dataclass(frozen=True)
class Field:
    """One control, and the TOML key it writes.

    Attributes:
        path: Dotted key, e.g. ``target.manoeuvre.amplitude_g``. This is the
            whole contract with the validator: get it wrong and the scenario is
            rejected as an unknown key, which a test relies on.
        kind: How to draw it — ``number``, ``integer``, ``choice``, ``toggle``,
            ``position`` (three metres) or ``velocity`` (speed and direction).
        when: ``(path, values)``. Shown only while that field holds one of them.
        blank_is_default: A zero means "say nothing and let the validator
            choose". Only for keys whose default is computed rather than fixed.
        required: The scenario cannot be read without it, so ``default`` is only
            a seed for the form rather than what omitting the key would mean.
            Everywhere else the two are the same thing, and a test insists on it.
    """

    path: str
    label: str
    kind: str
    default: Any
    group: str
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    when: tuple[str, tuple[str, ...]] | None = None
    blank_is_default: bool = False
    required: bool = False
    unit: str = ""


#: Display order of the panels.
GROUPS: tuple[str, ...] = (
    "Engagement",
    "Target",
    "Manoeuvre",
    "Guidance",
    "Seeker",
    "Estimator",
    "Missile",
    "Motor",
    "Airframe",
)

_MANOEUVRE = "target.manoeuvre.kind"
_LAW = "guidance.law"
_ESTIMATOR = "estimator.kind"

FIELDS: tuple[Field, ...] = (
    # -- Engagement -------------------------------------------------------
    Field(
        "duration",
        "Give up after",
        "number",
        30.0,
        "Engagement",
        minimum=1.0,
        maximum=120.0,
        step=1.0,
        unit="s",
        help="Simulated seconds before the run is abandoned.",
    ),
    Field(
        "lethal_radius",
        "Lethal radius",
        "number",
        5.0,
        "Engagement",
        minimum=0.0,
        maximum=50.0,
        step=0.5,
        unit="m",
        help="A closest approach inside this counts as a hit.",
    ),
    # -- Target -----------------------------------------------------------
    Field(
        "target.position",
        "Starting position",
        "position",
        [0.0, 6000.0, 1000.0],
        "Target",
        required=True,
        help="East, north, up — metres.",
    ),
    Field(
        "target.velocity",
        "Speed and direction",
        "velocity",
        [250.0, 0.0, 0.0],
        "Target",
        required=True,
        help="Heading is degrees clockwise from north; climb is degrees above the horizon.",
    ),
    # -- Manoeuvre --------------------------------------------------------
    Field(
        _MANOEUVRE,
        "Manoeuvre",
        "choice",
        "straight",
        "Manoeuvre",
        choices=("straight", "weave", "break_turn", "barrel_roll", "jink"),
        help="What the target does about being shot at.",
    ),
    Field(
        "target.manoeuvre.amplitude_g",
        "Amplitude",
        "number",
        6.0,
        "Manoeuvre",
        minimum=0.0,
        maximum=12.0,
        step=0.5,
        unit="g",
        when=(_MANOEUVRE, ("weave", "break_turn", "barrel_roll", "jink")),
    ),
    Field(
        "target.manoeuvre.period",
        "Period",
        "number",
        4.0,
        "Manoeuvre",
        minimum=0.5,
        maximum=20.0,
        step=0.5,
        unit="s",
        when=(_MANOEUVRE, ("weave", "barrel_roll")),
        help="Seconds for one full cycle.",
    ),
    Field(
        "target.manoeuvre.bank_deg",
        "Bank",
        "number",
        0.0,
        "Manoeuvre",
        minimum=-180.0,
        maximum=180.0,
        step=5.0,
        unit="°",
        when=(_MANOEUVRE, ("weave", "break_turn")),
        help="0 is a flat turn, 90 a pure pull-up, -90 a split-S.",
    ),
    Field(
        "target.manoeuvre.start_time",
        "Breaks at",
        "number",
        0.0,
        "Manoeuvre",
        minimum=0.0,
        maximum=30.0,
        step=0.5,
        unit="s",
        when=(_MANOEUVRE, ("break_turn",)),
        help="A late break is the hardest case: the missile is slowest then.",
    ),
    Field(
        "target.manoeuvre.interval",
        "Changes every",
        "number",
        1.5,
        "Manoeuvre",
        minimum=0.1,
        maximum=10.0,
        step=0.1,
        unit="s",
        when=(_MANOEUVRE, ("jink",)),
    ),
    Field(
        "target.manoeuvre.seed",
        "Jink seed",
        "integer",
        0,
        "Manoeuvre",
        minimum=0,
        maximum=9999,
        step=1,
        when=(_MANOEUVRE, ("jink",)),
        help="Same seed, same sequence of breaks.",
    ),
    # -- Guidance ---------------------------------------------------------
    Field(
        _LAW,
        "Guidance law",
        "choice",
        "pronav",
        "Guidance",
        choices=("pronav", "apn", "pursuit", "none"),
        help="apn adds a term for the target's own acceleration; pursuit is the baseline.",
    ),
    Field(
        "guidance.navigation_constant",
        "Navigation constant",
        "number",
        3.0,
        "Guidance",
        minimum=1.0,
        maximum=6.0,
        step=0.1,
        when=(_LAW, ("pronav", "apn")),
        help="3 is optimal against a non-manoeuvring target; 4-5 leads harder.",
    ),
    # -- Seeker -----------------------------------------------------------
    Field(
        "seeker.enabled",
        "Seeker",
        "toggle",
        True,
        "Seeker",
        help="Off gives the guidance law perfect information — the baseline.",
    ),
    Field(
        "seeker.angle_sigma",
        "Angle noise",
        "number",
        0.002,
        "Seeker",
        minimum=0.0,
        maximum=0.02,
        step=0.0005,
        unit="rad",
        when=("seeker.enabled", ("true",)),
        help="The dominant error. Cross-range error is range x sigma.",
    ),
    Field(
        "seeker.glint_sigma",
        "Glint",
        "number",
        1.5,
        "Seeker",
        minimum=0.0,
        maximum=10.0,
        step=0.1,
        unit="m",
        when=("seeker.enabled", ("true",)),
        help="A distance, so its angular effect grows as range falls. Sets the miss floor.",
    ),
    Field(
        "seeker.range_sigma",
        "Range noise",
        "number",
        3.0,
        "Seeker",
        minimum=0.0,
        maximum=50.0,
        step=0.5,
        unit="m",
        when=("seeker.enabled", ("true",)),
    ),
    Field(
        "seeker.range_rate_sigma",
        "Doppler noise",
        "number",
        0.4,
        "Seeker",
        minimum=0.0,
        maximum=10.0,
        step=0.1,
        unit="m/s",
        when=("seeker.enabled", ("true",)),
    ),
    Field(
        "seeker.gimbal_limit_deg",
        "Gimbal limit",
        "number",
        40.0,
        "Seeker",
        minimum=5.0,
        maximum=90.0,
        step=1.0,
        unit="°",
        when=("seeker.enabled", ("true",)),
        help="Off boresight beyond this and the target is lost.",
    ),
    Field(
        "seeker.latency_frames",
        "Latency",
        "integer",
        1,
        "Seeker",
        minimum=0,
        maximum=10,
        step=1,
        unit="frames",
        when=("seeker.enabled", ("true",)),
        help="Guidance cycles of processing delay. Small number, large effect.",
    ),
    Field(
        "seeker.detection_range",
        "Detection range",
        "number",
        12000.0,
        "Seeker",
        minimum=1000.0,
        maximum=40000.0,
        step=500.0,
        unit="m",
        when=("seeker.enabled", ("true",)),
        help="Reference range for the R^-4 signal law.",
    ),
    Field(
        "seeker.detection_threshold_db",
        "Detection threshold",
        "number",
        12.0,
        "Seeker",
        minimum=0.0,
        maximum=30.0,
        step=0.5,
        unit="dB",
        when=("seeker.enabled", ("true",)),
    ),
    Field(
        "seeker.snr_fluctuation_db",
        "Scintillation",
        "number",
        3.0,
        "Seeker",
        minimum=0.0,
        maximum=15.0,
        step=0.5,
        unit="dB",
        when=("seeker.enabled", ("true",)),
    ),
    # -- Estimator --------------------------------------------------------
    Field(
        _ESTIMATOR,
        "Estimator",
        "choice",
        "ekf",
        "Estimator",
        choices=("ekf", "alpha_beta", "none"),
        # With no seeker the guidance law is handed the truth and `build`
        # ignores whatever filter is configured, so offering one would imply it
        # was running. The command line makes the same distinction in words.
        when=("seeker.enabled", ("true",)),
        help="none is the naive Phase 4 behaviour: velocity by differencing noisy positions.",
    ),
    Field(
        "estimator.jerk_sigma",
        "Jerk sigma",
        "number",
        60.0,
        "Estimator",
        minimum=1.0,
        maximum=400.0,
        step=5.0,
        when=(_ESTIMATOR, ("ekf",)),
        help="How strongly the filter disbelieves its own constant-acceleration model.",
    ),
    Field(
        "estimator.initial_velocity_sigma",
        "Initial velocity sigma",
        "number",
        400.0,
        "Estimator",
        minimum=1.0,
        maximum=2000.0,
        step=25.0,
        when=(_ESTIMATOR, ("ekf",)),
    ),
    Field(
        "estimator.initial_acceleration_sigma",
        "Initial acceleration sigma",
        "number",
        100.0,
        "Estimator",
        minimum=1.0,
        maximum=1000.0,
        step=10.0,
        when=(_ESTIMATOR, ("ekf",)),
    ),
    Field(
        "estimator.alpha",
        "Alpha",
        "number",
        0.25,
        "Estimator",
        minimum=0.01,
        maximum=0.99,
        step=0.01,
        when=(_ESTIMATOR, ("alpha_beta",)),
        help="Position gain. Small is heavy smoothing, which also smooths away manoeuvres.",
    ),
    Field(
        "estimator.beta",
        "Beta",
        "number",
        0.0,
        "Estimator",
        minimum=0.0,
        maximum=1.99,
        step=0.01,
        when=(_ESTIMATOR, ("alpha_beta",)),
        blank_is_default=True,
        help="0 leaves it critically damped at alpha^2 / (2 - alpha).",
    ),
    # -- Missile ----------------------------------------------------------
    Field(
        "missile.position",
        "Launch position",
        "position",
        [0.0, 0.0, 1000.0],
        "Missile",
        required=True,
        help="East, north, up — metres.",
    ),
    Field(
        "missile.speed",
        "Launch speed",
        "number",
        60.0,
        "Missile",
        minimum=0.0,
        maximum=600.0,
        step=10.0,
        unit="m/s",
        help="Off the rail, aimed at the target's starting position.",
    ),
    Field(
        "missile.mass",
        "Launch mass",
        "number",
        85.0,
        "Missile",
        minimum=1.0,
        maximum=500.0,
        step=5.0,
        unit="kg",
    ),
    Field(
        "missile.autopilot_lag",
        "Autopilot lag",
        "number",
        0.20,
        "Missile",
        minimum=0.0,
        maximum=2.0,
        step=0.05,
        unit="s",
        help="First-order lag from commanded to achieved acceleration.",
    ),
    # -- Motor ------------------------------------------------------------
    Field(
        "missile.motor.boost_thrust",
        "Boost thrust",
        "number",
        20000.0,
        "Motor",
        minimum=0.0,
        maximum=100000.0,
        step=1000.0,
        unit="N",
    ),
    Field(
        "missile.motor.boost_duration",
        "Boost duration",
        "number",
        2.5,
        "Motor",
        minimum=0.0,
        maximum=30.0,
        step=0.5,
        unit="s",
    ),
    Field(
        "missile.motor.sustain_thrust",
        "Sustain thrust",
        "number",
        0.0,
        "Motor",
        minimum=0.0,
        maximum=50000.0,
        step=500.0,
        unit="N",
    ),
    Field(
        "missile.motor.sustain_duration",
        "Sustain duration",
        "number",
        0.0,
        "Motor",
        minimum=0.0,
        maximum=60.0,
        step=1.0,
        unit="s",
    ),
    Field(
        "missile.motor.specific_impulse",
        "Specific impulse",
        "number",
        240.0,
        "Motor",
        minimum=50.0,
        maximum=400.0,
        step=10.0,
        unit="s",
        help="Sets how fast propellant mass is consumed.",
    ),
    # -- Airframe ---------------------------------------------------------
    Field(
        "missile.aero.drag_coefficient",
        "Zero-lift drag",
        "number",
        0.30,
        "Airframe",
        minimum=0.05,
        maximum=1.5,
        step=0.01,
        help="Constant — no Mach dependence is modelled.",
    ),
    Field(
        "missile.aero.reference_area",
        "Reference area",
        "number",
        0.02,
        "Airframe",
        minimum=0.001,
        maximum=0.5,
        step=0.005,
        unit="m²",
    ),
    Field(
        "missile.aero.max_lateral_g",
        "Structural limit",
        "number",
        30.0,
        "Airframe",
        minimum=1.0,
        maximum=100.0,
        step=1.0,
        unit="g",
        help="Available g is the lesser of this and what dynamic pressure can produce.",
    ),
    Field(
        "missile.aero.max_lift_coefficient",
        "Max lift coefficient",
        "number",
        2.50,
        "Airframe",
        minimum=0.1,
        maximum=6.0,
        step=0.1,
    ),
    Field(
        "missile.aero.peak_lift_angle_deg",
        "Peak lift angle",
        "number",
        25.0,
        "Airframe",
        minimum=1.0,
        maximum=89.0,
        step=1.0,
        unit="°",
        help="With max lift, sets what turning costs: Cd = Cd0 + k*Cn^2.",
    ),
)


def as_json() -> list[dict[str, Any]]:
    """The table, as the page consumes it."""
    return [
        {
            "path": f.path,
            "label": f.label,
            "kind": f.kind,
            "default": f.default,
            "group": f.group,
            "help": f.help,
            "min": f.minimum,
            "max": f.maximum,
            "step": f.step,
            "choices": list(f.choices),
            "when": None if f.when is None else {"path": f.when[0], "values": list(f.when[1])},
            "blankIsDefault": f.blank_is_default,
            "required": f.required,
            "unit": f.unit,
        }
        for f in FIELDS
    ]


#: By dotted path, for the transitive visibility rule below.
_BY_PATH: dict[str, Field] = {entry.path: entry for entry in FIELDS}


def _shown(entry: Field, values: dict[str, Any]) -> bool:
    """Whether a field is active given the choices currently made.

    The same rule the page applies, so that what the form would send and what
    this builds are the same scenario.

    Transitive, and it has to be. The estimator's tuning is gated on which
    estimator, which is itself gated on there being a seeker at all — and with
    no seeker the guidance law is handed perfect information, so every one of
    those knobs is inert. A rule that only looked one level up would hide the
    estimator and go on offering its jerk sigma.
    """
    seen: set[str] = set()
    current_field: Field | None = entry
    while current_field is not None and current_field.when is not None:
        if current_field.path in seen:  # pragma: no cover - a cycle in the table
            msg = f"{entry.path} is gated in a loop"
            raise ValueError(msg)
        seen.add(current_field.path)

        controlling, allowed = current_field.when
        current = values.get(controlling, "")
        # Toggles are compared as the strings the table lists, so one rule
        # covers both a choice and a checkbox.
        text = str(current).lower() if isinstance(current, bool) else str(current)
        if text not in allowed:
            return False
        current_field = _BY_PATH.get(controlling)
    return True


def default_scenario(**choices: Any) -> dict[str, Any]:
    """Every active field at its default, as a nested dict.

    ``choices`` overrides controlling fields by dotted path, so
    ``default_scenario(**{"target.manoeuvre.kind": "jink"})`` gives the keys a
    jink accepts and none of the ones it rejects.

    Exists for the round-trip test, which proves that every ``path`` above is a
    key the validator actually knows. An unknown key is an error rather than a
    shrug, so a typo here fails loudly instead of quietly doing nothing.
    """
    values: dict[str, Any] = {entry.path: entry.default for entry in FIELDS}
    values.update(choices)

    data: dict[str, Any] = {}
    for entry in FIELDS:
        if not _shown(entry, values):
            continue
        value = values[entry.path]
        if entry.blank_is_default and not value:
            continue
        parts = entry.path.split(".")
        table = data
        for part in parts[:-1]:
            table = table.setdefault(part, {})
        table[parts[-1]] = value
    return data


#: Settings the validator accepts that the form deliberately does not offer,
#: and why. A test checks this against the annotated reference scenario, so a
#: setting added to the validator without a control here has to be declared
#: rather than quietly forgotten.
OMITTED: dict[str, str] = {
    "name": "the scenario's own name, not a physical setting",
    "description": "prose; edit it in the TOML pane",
    "velocity": "the launch speed control aims the missile at the target, and the "
    "target's velocity has a speed-and-direction control of its own",
}
