"""The engagement geometry as the guidance law sees it.

A :class:`Track` is the *only* thing a guidance law is ever given. In this phase
it is filled in from truth, so the numbers are perfect; from Phase 4 the same
object arrives from a noisy seeker and an estimator, and not one line of the
guidance code has to change. That is the whole reason this type exists rather
than passing the two entities around.

The derived quantities live here too — range, closing velocity and the
line-of-sight rotation vector — because every guidance law needs them and there
is exactly one correct way to compute each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

from interceptor.core.state import EntityState, Vector
from interceptor.core.world import Entity
from interceptor.sensing.filters import Estimator
from interceptor.sensing.seeker import Measurement, Seeker, relative_position_from

__all__ = ["FilteredTrack", "SeekerTrack", "Track", "TrackSource", "TruthTrack"]

_EPS = 1e-12


@dataclass(frozen=True)
class Track:
    """Estimated relative geometry between the missile and its target.

    Attributes:
        time: When this estimate applies, seconds.
        relative_position: ``p_target - p_missile`` in the world frame, metres.
        relative_velocity: ``v_target - v_missile`` in the world frame, m/s.
        target_acceleration: Estimated target acceleration, m/s^2. ``None``
            until an estimator that can produce it exists (Phase 5); augmented
            proportional navigation needs it.
        valid: ``False`` when there is no usable measurement — outside the
            seeker's field of view, below its detection threshold, or otherwise
            dropped. A dropout is an ordinary value flowing through the system,
            not an exception.
    """

    time: float
    relative_position: Vector
    relative_velocity: Vector
    target_acceleration: Vector | None = None
    valid: bool = True

    @property
    def range(self) -> float:
        """Distance to the target, metres."""
        return float(np.linalg.norm(self.relative_position))

    @property
    def line_of_sight(self) -> Vector:
        """Unit vector from the missile towards the target."""
        distance = self.range
        if distance < _EPS:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.relative_position / distance, dtype=np.float64)

    @property
    def closing_speed(self) -> float:
        """Rate at which range is shrinking, m/s. Positive while closing.

        This is ``V_c`` in the guidance literature: the negative of the range
        rate, so that a healthy intercept has a positive value.
        """
        distance = self.range
        if distance < _EPS:
            return 0.0
        return -float(np.dot(self.relative_position, self.relative_velocity)) / distance

    @property
    def los_rate_vector(self) -> Vector:
        """The line-of-sight rotation vector, rad/s.

        ``Omega = (r x v) / (r . r)``. Its magnitude is the line-of-sight rate
        that proportional navigation drives to zero, and its direction is the
        axis the sightline is rotating about. This is the single most important
        quantity in the whole project — everything the sensing chain does exists
        to estimate it well.
        """
        r = self.relative_position
        denominator = float(np.dot(r, r))
        if denominator < _EPS:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(np.cross(r, self.relative_velocity) / denominator, dtype=np.float64)

    @property
    def los_rate(self) -> float:
        """Magnitude of the line-of-sight rate, rad/s."""
        return float(np.linalg.norm(self.los_rate_vector))

    @property
    def time_to_go(self) -> float:
        """Crude time until closest approach, seconds.

        ``range / closing_speed``. Exact only for a constant closing velocity,
        which is close enough for a terminal engagement and is what the
        classical guidance derivations assume. Infinite when not closing.
        """
        closing = self.closing_speed
        if closing <= _EPS:
            return float("inf")
        return self.range / closing


class TrackSource(ABC):
    """Produces a :class:`Track` for the guidance law on demand.

    One implementation now (truth), and from Phase 4 a second one built from a
    seeker and a filter. The guidance law cannot tell them apart, which is what
    makes "run this law on perfect information" a one-line debugging move for
    the rest of the project.
    """

    @abstractmethod
    def update(self, t: float, missile: EntityState) -> Track:
        """Return the best current estimate of the engagement geometry."""


class TruthTrack(TrackSource):
    """A perfect track, read straight out of the world.

    No noise, no latency, no field of view, no dropouts. Useful for exactly two
    things: getting a guidance law working before the seeker exists, and
    afterwards as the baseline every degraded configuration is measured against.

    It holds the target *entity*, not a copy of its state. An entity's state is
    replaced by the integrator on every step, so keeping the entity means the
    track is always current; keeping a state would silently freeze the target at
    its starting position — a bug that produces a missile confidently
    intercepting empty air.
    """

    def __init__(self, target: Entity) -> None:
        self.target = target

    def update(self, t: float, missile: EntityState) -> Track:
        target = self.target.state
        return Track(
            time=t,
            relative_position=target.pos - missile.pos,
            relative_velocity=target.vel - missile.vel,
            valid=True,
        )


class SeekerTrack(TrackSource):
    """A track built from seeker measurements, with no estimator behind it.

    This is the naive implementation, and it is deliberately naive. It rebuilds
    the relative position from each measurement and then obtains relative
    velocity by *differencing successive positions* — which is the obvious thing
    to do and very nearly the worst.

    Differencing amplifies noise by one over the timestep. At 100 Hz that is a
    factor of a hundred: a 2 mrad angle error at 5 km is 10 m of cross-range
    error, and differencing two such errors 10 ms apart implies a relative
    velocity wrong by hundreds of metres per second. Proportional navigation
    then multiplies that by the navigation constant and the closing speed, and
    commands the airframe accordingly.

    The result is a missile that flails. That is the Phase 4 finding, and it is
    what the estimator in Phase 5 exists to fix — the same seeker, the same
    guidance law, with something sensible in between.

    Dropouts are reported as an invalid track and the guidance law is expected
    to coast. Nothing is extrapolated here; inventing data to cover a gap is the
    estimator's job, and it does not exist yet.
    """

    def __init__(self, seeker: Seeker, target: Entity) -> None:
        self.seeker = seeker
        self.target = target
        self.latest: Measurement | None = None
        self._previous_position: Vector | None = None
        self._previous_time: float | None = None
        self._previous_velocity: Vector = np.zeros(3, dtype=np.float64)

    def update(self, t: float, missile: EntityState) -> Track:
        measurement = self.seeker.measure(t, missile, self.target.state)
        self.latest = measurement

        if not measurement.valid:
            # Drop the differencing history: resuming after a gap with a stale
            # position would produce one enormous bogus velocity.
            self._previous_position = None
            self._previous_time = None
            return Track(
                time=t,
                relative_position=np.zeros(3, dtype=np.float64),
                relative_velocity=np.zeros(3, dtype=np.float64),
                valid=False,
            )

        position = relative_position_from(measurement, missile)

        velocity = self._previous_velocity
        if self._previous_position is not None and self._previous_time is not None:
            interval = t - self._previous_time
            if interval > _EPS:
                velocity = (position - self._previous_position) / interval

        self._previous_position = position
        self._previous_time = t
        self._previous_velocity = velocity

        return Track(
            time=t,
            relative_position=position,
            relative_velocity=velocity,
            valid=True,
        )


class FilteredTrack(TrackSource):
    """Seeker measurements passed through an estimator.

    The whole of Phase 5 in one class. Where :class:`SeekerTrack` differences
    successive positions and amplifies the noise a hundredfold, this predicts
    the target forward on a motion model and corrects that prediction with
    whatever the seeker managed to see.

    The ordering is the important part, and it is why the estimator interface
    splits prediction from correction. Every cycle, *predict* — unconditionally,
    measurement or not. Then correct, but only if there is something to correct
    with. A dropout therefore costs the track nothing but confidence: the filter
    keeps propagating and the guidance law keeps being served, where the naive
    version threw the track away and the missile coasted blind.

    Target acceleration is passed through to the track when the estimator can
    supply it, which is what augmented proportional navigation will consume in
    Phase 7. The alpha-beta filter reports zero, so APN degrades to PN behind
    it — correctly, and without the guidance law needing to know why.

    **Two clocks, and why keeping them apart matters.** A measurement is stamped
    with when it was *taken*, which under modelled latency is earlier than when
    it arrives. Those are different instants and this class is where they meet,
    so it is this class's job not to confuse them.

    Everything the estimator sees runs on measurement time. It is predicted
    forward to the measurement's epoch, not to now, and it is corrected using
    the missile's state *as it was at that epoch* — which is why a short history
    of missile states is kept. Feeding a filter a measurement taken 10 ms ago
    while telling it where the missile is now embeds the relative motion over
    that interval into the estimate as a standing bias: at 650 m/s of closing
    that is 6.5 m, several times the position uncertainty the filter reports,
    and it is invisible in miss distance because the filter still tracks well
    enough to hit. It shows up immediately in a NEES consistency check, which is
    how it was found.

    The track handed to the guidance law then runs on current time, extrapolated
    forward from the estimator's epoch by however stale the last measurement is.
    That extrapolation is deterministic and belongs here rather than inside a
    filter, whose job is to estimate the state at the instant it was observed.
    """

    def __init__(
        self, seeker: Seeker, target: Entity, estimator: Estimator, history: int = 16
    ) -> None:
        self.seeker = seeker
        self.target = target
        self.estimator = estimator
        self.latest: Measurement | None = None
        self.dropouts = 0
        #: The instant the estimator's state refers to — a measurement's epoch
        #: after a sighting, the current time after coasting through a dropout.
        self._state_time: float | None = None
        self._history: deque[tuple[float, EntityState]] = deque(maxlen=history)

    def _missile_at(self, when: float, fallback: EntityState) -> EntityState:
        """The recorded missile state nearest ``when``.

        Nearest rather than interpolated: the history is sampled at the guidance
        rate and measurements are stamped at those same instants, so the match
        is exact in practice and interpolation would only add a way to be
        subtly wrong.
        """
        best, smallest = fallback, float("inf")
        for time, state in self._history:
            gap = abs(time - when)
            if gap < smallest:
                best, smallest = state, gap
        return best

    def update(self, t: float, missile: EntityState) -> Track:
        self._history.append((t, missile))
        measurement = self.seeker.measure(t, missile, self.target.state)
        self.latest = measurement

        # Advance the estimator to the epoch its next evidence describes: the
        # measurement's own timestamp when there is one, otherwise all the way
        # to now, because coasting means propagating the model to the present.
        epoch = measurement.time if measurement.valid else t
        interval = 0.0 if self._state_time is None else max(epoch - self._state_time, 0.0)
        self.estimator.predict(interval)
        self._state_time = epoch

        if measurement.valid:
            self.estimator.correct(measurement, self._missile_at(measurement.time, missile))
        else:
            self.dropouts += 1

        estimate = self.estimator.estimate()
        if estimate is None:
            # Nothing seen yet. Coasting straight is the honest response to
            # having no information at all.
            return Track(
                time=t,
                relative_position=np.zeros(3, dtype=np.float64),
                relative_velocity=np.zeros(3, dtype=np.float64),
                valid=False,
            )

        # Carry the estimate forward from its own epoch to now. Zero after a
        # dropout, one frame behind a latent seeker.
        lag = max(t - self._state_time, 0.0)
        position = (
            estimate.position + estimate.velocity * lag + 0.5 * estimate.acceleration * lag**2
        )
        velocity = estimate.velocity + estimate.acceleration * lag

        return Track(
            time=t,
            relative_position=position - missile.pos,
            relative_velocity=velocity - missile.vel,
            target_acceleration=estimate.acceleration,
            valid=True,
        )
