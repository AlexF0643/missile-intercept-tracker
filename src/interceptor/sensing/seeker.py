"""The seeker: the missile's only, deliberately lossy, window onto the truth.

Everything up to Phase 3 fed the guidance law perfect information. This module
takes that away. It computes the true geometry, then degrades it the way a real
radar seeker does — and the interesting part of the project starts here, because
a guidance law that was flawless on truth data now has to work with measurements
that are noisy, late, and sometimes absent entirely.

**The measurement chain, per cycle.** Point the seeker along the missile's
velocity; work out where the target is relative to that; check it is within the
gimbal's mechanical reach; decide whether the return is strong enough to detect
at all; then corrupt range, the two angles and range-rate with their respective
errors; then hold the result back by one frame to model processing latency.

**Why the errors behave as they do.** Angle noise is the dominant term and its
*cross-range* effect shrinks as you close, because an angular error subtends
less distance at shorter range. Glint runs the other way: it is a wander of the
target's apparent centre measured in metres, so its *angular* effect grows as
range falls. Those two crossing over is why terminal miss distance has a floor
that no amount of filtering removes.

**Detection.** Received power falls as the fourth power of range, so the
signal-to-noise ratio climbs 12 dB every time the range halves. Around the
nominal detection range the return flickers above and below the threshold as
the target's radar cross-section scintillates, which produces exactly the
dropout behaviour a real system has to cope with — solid lock up close, an
intermittent one far out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from interceptor.core.frames import (
    az_el_from_frd,
    body_axes,
    body_to_world,
    frd_from_az_el,
    world_to_body,
)
from interceptor.core.state import EntityState, Vector

__all__ = [
    "GeometricSeeker",
    "Measurement",
    "Seeker",
    "SeekerConfig",
    "relative_position_from",
]

_EPS: Final = 1e-9


@dataclass(frozen=True)
class Measurement:
    """One seeker report, in the missile's body frame.

    Attributes:
        time: When the measurement was taken, seconds. With latency modelled,
            this is *earlier* than the time it is delivered — which is the
            whole point of recording it.
        valid: Whether there is a usable return at all.
        reason: Why not, when ``valid`` is False. Empty otherwise.
        range: Measured range to the target, metres.
        azimuth: Off-boresight angle to the right, radians.
        elevation: Off-boresight angle upward, radians.
        range_rate: Measured rate of change of range, m/s. Negative while
            closing, since range is shrinking.
        snr_db: Signal-to-noise ratio of this return, decibels. Recorded for
            diagnostics; the guidance chain never looks at it.
    """

    time: float
    valid: bool
    reason: str = ""
    range: float = 0.0
    azimuth: float = 0.0
    elevation: float = 0.0
    range_rate: float = 0.0
    snr_db: float = 0.0

    @classmethod
    def dropout(cls, time: float, reason: str, snr_db: float = 0.0) -> Measurement:
        """A cycle with no usable return."""
        return cls(time=time, valid=False, reason=reason, snr_db=snr_db)


@dataclass(frozen=True)
class SeekerConfig:
    """The seeker's error budget. Every field is a knob worth sweeping.

    Defaults are representative of a semi-active radar seeker. They are not
    taken from any particular system; they are round numbers of the right order,
    chosen so the trade-offs behave realistically.

    Attributes:
        range_sigma: Range measurement noise, metres. Barely matters —
            proportional navigation cares about angles, not range.
        angle_sigma: Angle noise on each axis, radians. 2 mrad is 20 m of
            cross-range error at 10 km, and 0.2 m at 100 m. The dominant term.
        range_rate_sigma: Doppler noise, m/s. Doppler is precise, which is why
            closing velocity is the one quantity a seeker measures well.
        glint_sigma: Wander of the target's apparent centre, metres. Constant in
            metres means growing in angle as range falls — the floor under
            terminal miss distance.
        gimbal_limit: How far off the missile's nose the seeker head can look,
            radians. Beyond it there is no measurement at all. A high-crossing
            target can walk out of this and never come back.
        detection_range: Range at which the mean signal-to-noise ratio equals
            the detection threshold, metres. Beyond it, returns are
            intermittent; well inside it, lock is solid.
        snr_fluctuation_db: Standard deviation of the return's strength,
            decibels, modelling radar-cross-section scintillation.
        latency_frames: How many guidance cycles the measurement is held back.
            One frame at 100 Hz is 10 ms — a small number with a large effect,
            because the command that matters most is the last one.
    """

    range_sigma: float = 3.0
    angle_sigma: float = 2.0e-3
    range_rate_sigma: float = 0.4
    glint_sigma: float = 1.5
    gimbal_limit: float = np.deg2rad(40.0)
    detection_range: float = 12_000.0
    snr_fluctuation_db: float = 3.0
    latency_frames: int = 1

    #: Threshold the return must beat to count as a detection, decibels.
    detection_threshold_db: float = 12.0

    def __post_init__(self) -> None:
        if self.latency_frames < 0:
            msg = f"latency_frames cannot be negative, got {self.latency_frames}"
            raise ValueError(msg)
        if self.detection_range <= 0.0:
            msg = f"detection_range must be positive, got {self.detection_range}"
            raise ValueError(msg)

    @classmethod
    def perfect(cls) -> SeekerConfig:
        """A seeker with no errors at all.

        Not physically meaningful, but invaluable as a control: running the full
        seeker path with every error switched off should reproduce the
        truth-data results of Phase 3. If it does not, the bug is in the
        measurement chain rather than in the noise.
        """
        return cls(
            range_sigma=0.0,
            angle_sigma=0.0,
            range_rate_sigma=0.0,
            glint_sigma=0.0,
            gimbal_limit=np.pi,
            detection_range=1e9,
            snr_fluctuation_db=0.0,
            latency_frames=0,
        )

    def scaled(self, factor: float) -> SeekerConfig:
        """A copy with every *noise* term multiplied by ``factor``.

        Gating and latency are left alone, so a sweep over ``factor`` varies
        measurement quality without also changing when the seeker can see at
        all — which keeps the resulting trend attributable to one cause.
        """
        return SeekerConfig(
            range_sigma=self.range_sigma * factor,
            angle_sigma=self.angle_sigma * factor,
            range_rate_sigma=self.range_rate_sigma * factor,
            glint_sigma=self.glint_sigma * factor,
            gimbal_limit=self.gimbal_limit,
            detection_range=self.detection_range,
            snr_fluctuation_db=self.snr_fluctuation_db,
            latency_frames=self.latency_frames,
            detection_threshold_db=self.detection_threshold_db,
        )


class Seeker(ABC):
    """Produces a :class:`Measurement` from the true geometry."""

    @abstractmethod
    def measure(self, t: float, missile: EntityState, target: EntityState) -> Measurement:
        """Look for the target and report what was seen, if anything."""


@dataclass
class GeometricSeeker(Seeker):
    """Truth geometry, degraded.

    Args:
        config: The error budget.
        rng: Seeded generator. Owned by the scenario so that a run is
            reproducible bit for bit — which is what makes a Monte Carlo sweep
            a measurement rather than an anecdote.
    """

    config: SeekerConfig
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _pipeline: deque[Measurement] = field(default_factory=deque, init=False, repr=False)

    def measure(self, t: float, missile: EntityState, target: EntityState) -> Measurement:
        fresh = self._observe(t, missile, target)

        if self.config.latency_frames == 0:
            return fresh

        # Processing delay: the answer delivered now was computed from what the
        # world looked like a frame ago. Prime the pipeline with dropouts so the
        # first cycles report "no data yet" rather than inventing a return.
        self._pipeline.append(fresh)
        while len(self._pipeline) <= self.config.latency_frames:
            self._pipeline.appendleft(Measurement.dropout(t, "seeker still settling"))
        while len(self._pipeline) > self.config.latency_frames + 1:
            self._pipeline.popleft()
        return self._pipeline[0]

    def _observe(self, t: float, missile: EntityState, target: EntityState) -> Measurement:
        """One un-delayed look at the target."""
        relative = target.pos - missile.pos
        true_range = float(np.linalg.norm(relative))
        if true_range < _EPS:
            return Measurement.dropout(t, "target coincident with missile")

        # Glint: the apparent centre of a complex target wanders by a distance,
        # not by an angle. Applied to the position before angles are taken, so
        # its angular effect grows automatically as the range falls.
        apparent = relative
        if self.config.glint_sigma > 0.0:
            apparent = relative + self.rng.normal(0.0, self.config.glint_sigma, size=3)

        if missile.speed < _EPS:
            return Measurement.dropout(t, "missile has no heading to point along")
        axes = body_axes(missile.vel)
        body = world_to_body(axes, apparent)

        measured_range, azimuth, elevation = az_el_from_frd(body)
        if measured_range < _EPS:
            return Measurement.dropout(t, "target coincident with missile")

        off_boresight = float(np.arccos(np.clip(body[0] / measured_range, -1.0, 1.0)))
        if off_boresight > self.config.gimbal_limit:
            return Measurement.dropout(t, "beyond the gimbal limit")

        snr_db = self._signal_to_noise(true_range)
        if snr_db < self.config.detection_threshold_db:
            return Measurement.dropout(t, "below the detection threshold", snr_db)

        relative_velocity = target.vel - missile.vel
        true_range_rate = float(np.dot(relative, relative_velocity)) / true_range

        return Measurement(
            time=t,
            valid=True,
            range=measured_range + self._noise(self.config.range_sigma),
            azimuth=azimuth + self._noise(self.config.angle_sigma),
            elevation=elevation + self._noise(self.config.angle_sigma),
            range_rate=true_range_rate + self._noise(self.config.range_rate_sigma),
            snr_db=snr_db,
        )

    def _signal_to_noise(self, true_range: float) -> float:
        """Return strength in decibels, from the radar equation's ``R^-4`` law.

        Expressed relative to the configured detection range, so ``SNR`` equals
        the threshold exactly at that range and rises 12 dB for every halving
        of range thereafter.
        """
        mean_db = self.config.detection_threshold_db + 40.0 * float(
            np.log10(self.config.detection_range / max(true_range, _EPS))
        )
        return mean_db + self._noise(self.config.snr_fluctuation_db)

    def _noise(self, sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, sigma))


def relative_position_from(measurement: Measurement, missile: EntityState) -> Vector:
    """Turn a body-frame measurement back into a world-frame offset.

    This is the "work out the location relative to the missile" step: take the
    measured range and two angles, rebuild the Cartesian offset in the body
    frame, and rotate it into world coordinates using the missile's attitude.
    It is the exact inverse of what the seeker did, which is why the frame
    round-trip test written back in Phase 1 matters so much — a sign error here
    produces a smooth, plausible and completely wrong trajectory.
    """
    body = frd_from_az_el(measurement.range, measurement.azimuth, measurement.elevation)
    return body_to_world(body_axes(missile.vel), body)
