"""Where every constant came from, and how much it could be wrong by.

This file exists because of a mistake. Augmented proportional navigation was
found to miss a barrel roll by 469 m, and that was written up as its
constant-acceleration premise failing. It was not. Doubling one number —
``max_lift_coefficient``, chosen because 2.5 sounded like a plausible value for
a finned airframe — turned the same engagement into a 2 cm intercept. An uncited
constant was the difference between a guidance law being excellent and being
catastrophic, and nothing in the project said so.

So each constant below is classified by *where it came from*, which is a
different question from what it is:

``defined``
    Fixed by convention or standards body. Not uncertain in any useful sense.
``derived``
    Follows from other declared quantities. Cannot be varied independently
    without contradicting something.
``design``
    A choice that *defines this notional missile* rather than a fact about the
    world. A different value describes a different weapon, not a better guess at
    this one. Not uncertainty — configuration.
``chosen``
    Picked because it was plausible. **These are the uncited ones**, and they are
    what the Monte Carlo samples.

No entry here claims a citation, because none was consulted. Where a real
programme would open a wind-tunnel database or a radar link budget, this says
"chosen" and gives the range within which the answer plausibly lies. That is
less satisfying than a reference and considerably more honest than a reference
that was not read.

**The ranges are judgement, not data.** They are wide enough to contain what the
true value plausibly is for a body of this class, and stating them is what makes
the Monte Carlo able to say "somewhere between these two probabilities" instead
of a single number with false precision. Sampling is log-uniform across each
range: every constant here is a positive scale parameter, uncertainty in such
things is multiplicative, and a log-uniform draw is the flattest honest way of
saying "anywhere in here, and we do not know where".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

__all__ = [
    "CONSTANTS",
    "SAMPLED",
    "Constant",
    "Provenance",
    "apply",
    "draw",
    "nominal",
]

Provenance = Literal["defined", "derived", "design", "chosen"]


@dataclass(frozen=True)
class Constant:
    """One number in the model, and an account of where it came from.

    Attributes:
        path: Dotted scenario key, or a bare name for something that is not
            configurable per scenario.
        low, high: The range within which the true value plausibly lies, for
            ``chosen`` constants. ``None`` for everything else — a defined
            constant has no range, and varying a design choice describes a
            different missile rather than the same one more honestly.
    """

    path: str
    value: float
    unit: str
    provenance: Provenance
    why: str
    low: float | None = None
    high: float | None = None

    @property
    def sampled(self) -> bool:
        """Whether the Monte Carlo varies this one."""
        return self.provenance == "chosen" and self.low is not None and self.high is not None


CONSTANTS: Final[tuple[Constant, ...]] = (
    # -- defined -----------------------------------------------------------
    Constant(
        "STANDARD_GRAVITY",
        9.80665,
        "m/s^2",
        "defined",
        "The standard acceleration of gravity, fixed by the CGPM. Not a "
        "measurement of anywhere in particular.",
    ),
    Constant(
        "sea_level_density",
        1.225,
        "kg/m^3",
        "defined",
        "ISA sea-level density, which is a definition rather than an "
        "observation — the standard atmosphere is a convention.",
    ),
    # -- conventional modelling choices, recorded as design ----------------
    Constant(
        "scale_height",
        8500.0,
        "m",
        "design",
        "Exponential atmosphere, rho = 1.225 * exp(-h / 8500). The standard "
        "first approximation; the real atmosphere is not exponential, and this "
        "model does not go high enough for that to matter.",
    ),
    Constant(
        "seeker.latency_frames",
        1.0,
        "frames",
        "design",
        "One guidance cycle of processing delay. A modelling decision about "
        "what to represent, not an estimate of a real seeker's latency.",
    ),
    # -- derived -----------------------------------------------------------
    Constant(
        "induced_drag_factor",
        0.175,
        "-",
        "derived",
        "k = 1 / Cn_alpha, and the lift-curve slope is max_lift_coefficient "
        "over peak_lift_angle_deg. Follows entirely from those two, so it moves "
        "when they are sampled and must never be sampled itself.",
    ),
    Constant(
        "guidance.navigation_constant",
        3.0,
        "-",
        "derived",
        "The optimal navigation constant against a non-manoeuvring target under "
        "a minimum-effort criterion. A result, not a tuning knob — though 3 to "
        "5 is the practical range and the scenario file may say otherwise.",
    ),
    # -- design: this notional missile, not an uncertain fact ---------------
    Constant("missile.mass", 85.0, "kg", "design", "Launch mass of the round being modelled."),
    Constant(
        "missile.motor.boost_thrust",
        20000.0,
        "N",
        "design",
        "Motor size. Choosing differently describes a different missile.",
    ),
    Constant("missile.motor.boost_duration", 2.5, "s", "design", "Burn time of that same motor."),
    Constant(
        "missile.aero.max_lateral_g",
        30.0,
        "g",
        "design",
        "Structural limit. A stated requirement of the airframe rather than a "
        "measured property of one.",
    ),
    Constant(
        "missile.speed",
        60.0,
        "m/s",
        "design",
        "Speed off the rail, aimed at the target's starting position. A launch "
        "condition — part of the question being asked.",
    ),
    Constant(
        "missile.motor.sustain_thrust",
        0.0,
        "N",
        "design",
        "Zero: the modelled round is boost-only. Setting it describes a "
        "boost-sustain motor, which is a different missile.",
    ),
    Constant(
        "missile.motor.sustain_duration",
        0.0,
        "s",
        "design",
        "Likewise zero, and likewise a description rather than an estimate.",
    ),
    Constant(
        "seeker.gimbal_limit_deg",
        40.0,
        "deg",
        "design",
        "How far off boresight the seeker head can look. A specification.",
    ),
    Constant(
        "lethal_radius",
        5.0,
        "m",
        "design",
        "What counts as a hit. Defines the question being asked, so varying it "
        "would change the question rather than the answer.",
    ),
    # -- chosen: plausible, uncited, and sampled ---------------------------
    Constant(
        "missile.aero.drag_coefficient",
        0.30,
        "-",
        "chosen",
        "Zero-lift drag of a slender finned body. The single most influential "
        "number in the model — a sensitivity check moved the miss distance by "
        "-41% / +189% across a plus-or-minus 40% swing — and it is a guess. A "
        "real value is Mach-dependent and comes from a drag build-up or a "
        "tunnel; this is one number standing in for a curve.",
        low=0.18,
        high=0.55,
    ),
    Constant(
        "missile.aero.max_lift_coefficient",
        2.50,
        "-",
        "chosen",
        "Peak normal-force coefficient. The constant that decides whether "
        "augmented pronav is excellent or catastrophic, which is how this file "
        "came to exist. Body-plus-fin lift at high incidence varies enormously "
        "with configuration.",
        low=1.5,
        high=4.5,
    ),
    Constant(
        "missile.aero.peak_lift_angle_deg",
        25.0,
        "deg",
        "chosen",
        "The incidence at which that peak lift is reached, which with it sets "
        "the lift-curve slope and hence what turning costs.",
        low=15.0,
        high=35.0,
    ),
    Constant(
        "missile.aero.reference_area",
        0.02,
        "m^2",
        "chosen",
        "Body cross-section. Nearly derived — 0.02 m^2 is a 160 mm diameter — "
        "so the uncertainty here is really about what diameter this notional "
        "round is, and the range below spans roughly 140 to 225 mm.",
        low=0.015,
        high=0.040,
    ),
    Constant(
        "missile.motor.specific_impulse",
        240.0,
        "s",
        "chosen",
        "Sets how fast propellant mass is consumed for a given thrust. Solid "
        "propellants sit in a well-known band and this is inside it, but the "
        "specific figure was not taken from anywhere.",
        low=200.0,
        high=270.0,
    ),
    Constant(
        "missile.autopilot_lag",
        0.20,
        "s",
        "chosen",
        "First-order lag from commanded to achieved acceleration, standing in "
        "for fin actuation and airframe response that this 3-DOF model does not "
        "represent.",
        low=0.08,
        high=0.45,
    ),
    Constant(
        "seeker.angle_sigma",
        0.002,
        "rad",
        "chosen",
        "Angular measurement noise, the dominant seeker error. Cross-range "
        "error is range times this, so 2 mrad is 10 m at 5 km.",
        low=0.0008,
        high=0.005,
    ),
    Constant(
        "seeker.glint_sigma",
        1.5,
        "m",
        "chosen",
        "Apparent wander of the scattering centre across the target. A "
        "*distance*, so its angular effect grows as the range falls, which is "
        "what puts a floor under terminal miss distance. Its scale is set by "
        "how big the target is, and this assumes a fighter-sized one.",
        low=0.5,
        high=4.0,
    ),
    Constant(
        "seeker.detection_range",
        12000.0,
        "m",
        "chosen",
        "Reference range for the R^-4 signal law — the range at which the "
        "return sits at the threshold. Stands in for a whole link budget: "
        "transmit power, aperture, target cross-section and receiver noise, "
        "none of which this model represents.",
        low=6000.0,
        high=25000.0,
    ),
    Constant(
        "seeker.detection_threshold_db",
        12.0,
        "dB",
        "chosen",
        "Signal-to-noise ratio below which the target is not seen. A plausible "
        "figure for a detector holding a low false-alarm rate, not a derived one.",
        low=8.0,
        high=17.0,
    ),
    Constant(
        "seeker.snr_fluctuation_db",
        3.0,
        "dB",
        "chosen",
        "Scintillation of the return as the target's aspect changes. Real, "
        "well documented in general, and this particular number is a guess.",
        low=1.0,
        high=6.0,
    ),
    Constant(
        "seeker.range_sigma",
        3.0,
        "m",
        "chosen",
        "Range measurement noise. Barely matters to a law that steers on "
        "bearing, which the sensitivity results ought to confirm.",
        low=1.0,
        high=15.0,
    ),
    Constant(
        "seeker.range_rate_sigma",
        0.4,
        "m/s",
        "chosen",
        "Doppler noise. The precise channel, and the one the filter leans on for velocity.",
        low=0.1,
        high=2.0,
    ),
)

#: The ones the Monte Carlo varies: every constant admitted to be a guess.
SAMPLED: Final[tuple[Constant, ...]] = tuple(c for c in CONSTANTS if c.sampled)


def nominal() -> dict[str, float]:
    """Every sampled constant at the value the shipped scenarios use."""
    return {c.path: c.value for c in SAMPLED}


def draw(rng: np.random.Generator) -> dict[str, float]:
    """One plausible set of constants.

    Log-uniform across each declared range, independently. Independence is a
    simplification and worth naming: ``max_lift_coefficient`` and
    ``peak_lift_angle_deg`` describe the same lift curve and a real airframe
    would not vary them separately. Modelling that correlation would need a
    source for it, and inventing a correlation is exactly the sort of unearned
    precision this file exists to avoid — so they are drawn independently, which
    spreads the result slightly wider than the truth.
    """
    return {
        c.path: float(np.exp(rng.uniform(np.log(c.low), np.log(c.high))))
        for c in SAMPLED
        if c.low is not None and c.high is not None
    }


def apply(scenario: dict[str, Any], overrides: dict[str, float]) -> dict[str, Any]:
    """A copy of ``scenario`` with the drawn constants written into it.

    Returns a new nested dictionary rather than editing in place, because a
    Monte Carlo hands the same base scenario to a thousand draws and one shared
    mutable table would make every result depend on the order they ran in.
    """
    updated = copy.deepcopy(scenario)
    for path, value in overrides.items():
        parts = path.split(".")
        table = updated
        for part in parts[:-1]:
            table = table.setdefault(part, {})
        table[parts[-1]] = value
    return updated
