"""Estimators: turning a stream of noisy measurements into a usable track.

Phase 4 established that a realistic seeker breaks the intercept — not because
the measurements are bad, but because differencing two of them 10 ms apart to
get velocity multiplies the error by a hundred. This module is the answer.

**What is estimated, and why it is the target's absolute state.** Both filters
here estimate the *target's* position, velocity and acceleration in the world
frame, not the relative state. That choice matters. Relative velocity changes
because of the target's acceleration *and* the missile's own, so a filter on the
relative state has to be told what the missile just did, and any error in that
becomes an error in the track. The missile's own state is known — in a real
system from its inertial unit, here exactly — so estimating the target alone and
subtracting is both simpler and better conditioned. It also means the filter's
model, "the target holds its acceleration", is a statement about the target
rather than about a coupled pair.

**The two filters.** The alpha-beta filter is a Kalman filter with the gains
frozen rather than computed: two constants, no covariance, about thirty lines.
It is worth building first because when it misbehaves the reason is visible.
The extended Kalman filter computes its gains from a covariance it propagates,
so it adapts — trusting measurements while uncertain and its model once
confident — and it estimates target acceleration, which augmented proportional
navigation needs.

**The tension neither filter escapes.** Filtering harder gives less noise and
more lag. Against a weaving target, the smoothing that removes measurement
noise also removes the manoeuvre signal. There is no setting that wins both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import numpy as np

from interceptor.core.frames import az_el_from_frd, body_axes, world_to_body
from interceptor.core.state import EntityState, Vector, magnitude
from interceptor.sensing.seeker import Measurement, SeekerConfig, relative_position_from

__all__ = [
    "AlphaBeta",
    "Estimator",
    "ExtendedKalman",
    "TargetEstimate",
]

_EPS: Final = 1e-9


@dataclass(frozen=True)
class TargetEstimate:
    """What an estimator currently believes about the target, in the world frame."""

    position: Vector
    velocity: Vector
    acceleration: Vector
    covariance: Vector | None = None
    """State covariance, when the estimator maintains one. ``None`` for
    fixed-gain filters, which have no notion of their own uncertainty."""


class Estimator(ABC):
    """Maintains a running estimate of the target's state.

    The lifecycle is deliberately split. :meth:`predict` always runs, whether or
    not there is a measurement; :meth:`correct` runs only when there is one.
    That split is exactly what coasting through a dropout means — the filter
    keeps propagating its model forward and simply has nothing to correct it
    with, which is the right behaviour and impossible to express if prediction
    and correction are one call.
    """

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Propagate the estimate forward by ``dt`` seconds."""

    @abstractmethod
    def correct(self, measurement: Measurement, missile: EntityState) -> None:
        """Fold in one valid measurement."""

    @abstractmethod
    def estimate(self) -> TargetEstimate | None:
        """The current estimate, or ``None`` before the first measurement."""

    @property
    def name(self) -> str:
        return type(self).__name__


class AlphaBeta(Estimator):
    """Fixed-gain position and velocity filter.

    Predicts the target forward on its current velocity, compares that with the
    measurement, and corrects position by ``alpha`` of the discrepancy and
    velocity by ``beta / dt`` of it. No covariance, no matrices, two constants.

    Args:
        alpha: Position gain, in (0, 1). Larger trusts the measurement more.
        beta: Velocity gain. Defaults to the critically damped relation
            ``beta = alpha^2 / (2 - alpha)``, which is the value that gives the
            fastest response without the estimate ringing after a step change.
            Set it explicitly to explore the trade deliberately.

    It cannot estimate acceleration — it has no state for it — so it reports
    zero, and augmented proportional navigation degrades to plain PN behind it.
    Tracking a manoeuvring target with this filter leaves a systematic lag that
    no choice of gains removes, which is the argument for the EKF.
    """

    def __init__(self, alpha: float = 0.25, beta: float | None = None) -> None:
        if not 0.0 < alpha < 1.0:
            msg = f"alpha must be in (0, 1), got {alpha}"
            raise ValueError(msg)
        self.alpha = alpha
        self.beta = beta if beta is not None else alpha * alpha / (2.0 - alpha)
        if not 0.0 < self.beta < 2.0:
            msg = f"beta must be in (0, 2), got {self.beta}"
            raise ValueError(msg)

        self._position: Vector | None = None
        self._velocity: Vector = np.zeros(3, dtype=np.float64)
        self._last_time: float = 0.0

    def predict(self, dt: float) -> None:
        if self._position is None:
            return
        self._position = self._position + self._velocity * dt

    def correct(self, measurement: Measurement, missile: EntityState) -> None:
        observed = missile.pos + relative_position_from(measurement, missile)

        if self._position is None:
            # First sighting: take the measurement at face value and admit to
            # knowing nothing about velocity yet.
            self._position = observed
            self._velocity = np.zeros(3, dtype=np.float64)
            self._last_time = measurement.time
            return

        interval = measurement.time - self._last_time
        self._last_time = measurement.time
        if interval < _EPS:
            return

        residual = observed - self._position
        self._position = self._position + self.alpha * residual
        self._velocity = self._velocity + (self.beta / interval) * residual

    def estimate(self) -> TargetEstimate | None:
        if self._position is None:
            return None
        return TargetEstimate(
            position=self._position,
            velocity=self._velocity,
            acceleration=np.zeros(3, dtype=np.float64),
        )

    @property
    def name(self) -> str:
        return f"AlphaBeta (a={self.alpha:g})"


class ExtendedKalman(Estimator):
    """Nine-state constant-acceleration filter on the target's world-frame state.

    State is ``[position, velocity, acceleration]``, nine elements. The process
    model says the target holds its acceleration; the process noise says it
    does not really, and ``jerk_sigma`` is how strongly that is disbelieved.

    The measurement is what the seeker actually reports — range, azimuth,
    elevation and range-rate in the missile's body frame — rather than a
    position reconstructed from it. That keeps the precise Doppler measurement
    in the filter as a direct constraint on velocity, which is the single most
    useful thing a radar seeker provides, and it is what makes this *extended*:
    the map from state to measurement is nonlinear, so it has to be linearised
    at each step.

    Args:
        seeker: The error budget, so the filter knows how much to trust each
            measurement. Using the same numbers the seeker corrupts with is not
            cheating — a real system is designed around its own sensor's
            specification.
        jerk_sigma: Standard deviation of the target's jerk, m/s^3. This is the
            main tuning knob. Too small and the filter refuses to believe the
            target manoeuvred; too large and it chases noise.
        initial_velocity_sigma: Initial uncertainty in target velocity, m/s.
        initial_acceleration_sigma: Initial uncertainty in target acceleration.
    """

    def __init__(
        self,
        seeker: SeekerConfig | None = None,
        *,
        jerk_sigma: float = 60.0,
        initial_velocity_sigma: float = 400.0,
        initial_acceleration_sigma: float = 100.0,
    ) -> None:
        self.seeker = seeker if seeker is not None else SeekerConfig()
        self.jerk_sigma = jerk_sigma
        self.initial_velocity_sigma = initial_velocity_sigma
        self.initial_acceleration_sigma = initial_acceleration_sigma

        self._x: Vector | None = None
        self._P: Vector = np.eye(9, dtype=np.float64)
        self._last_time: float = 0.0

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    @staticmethod
    def _transition(dt: float) -> Vector:
        """Constant-acceleration state transition."""
        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * (0.5 * dt * dt)
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _process_noise(self, dt: float) -> Vector:
        """Continuous white-noise-jerk process noise, discretised.

        The standard result for a constant-acceleration model driven by white
        jerk. Note the coupling between blocks: a jerk that moves acceleration
        also moves velocity and position, and pretending otherwise makes the
        filter overconfident in exactly the situation it should be least sure
        about.
        """
        q = self.jerk_sigma * self.jerk_sigma
        t2, t3, t4, t5 = dt * dt, dt**3, dt**4, dt**5
        block = np.array(
            [
                [t5 / 20.0, t4 / 8.0, t3 / 6.0],
                [t4 / 8.0, t3 / 3.0, t2 / 2.0],
                [t3 / 6.0, t2 / 2.0, dt],
            ],
            dtype=np.float64,
        )
        # The state is ordered [p(3), v(3), a(3)], so the axes are strided: the
        # x-components sit at indices 0, 3, 6. The 3x3 kinematic block above
        # therefore scatters across the matrix rather than sitting in a corner
        # of it, once per spatial axis, with no coupling between axes.
        Q = np.zeros((9, 9), dtype=np.float64)
        for axis in range(3):
            for i in range(3):
                for j in range(3):
                    Q[i * 3 + axis, j * 3 + axis] = block[i, j] * q
        return Q

    def _measurement(self, x: Vector, missile: EntityState) -> Vector:
        """Predicted ``[range, azimuth, elevation, range_rate]`` for a state."""
        relative = x[0:3] - missile.pos
        relative_velocity = x[3:6] - missile.vel

        distance = magnitude(relative)
        if distance < _EPS:
            return np.zeros(4, dtype=np.float64)

        body = world_to_body(body_axes(missile.vel), relative)
        _, azimuth, elevation = az_el_from_frd(body)
        range_rate = float(np.dot(relative, relative_velocity)) / distance
        return np.array([distance, azimuth, elevation, range_rate], dtype=np.float64)

    def _jacobian_numeric(self, missile: EntityState) -> Vector:
        """Linearise the measurement model by central differences.

        This was the only implementation for five phases, and the argument for
        it was sound at the time: the analytical Jacobian of range, two
        body-frame angles and range-rate with respect to nine states is a page
        of algebra with a dozen chances to drop a sign, and a sign error there
        produces a filter that diverges slowly enough to look like a tuning
        problem. Eighteen evaluations of a twenty-flop function per cycle cost
        nothing when a study is six runs.

        A Monte Carlo is not six runs, and profiling put this at forty per cent
        of an engagement. So :meth:`_jacobian` now does the algebra — and this
        stays, as the thing that algebra is checked against. The concern was
        never that hand-derived partials are slow; it was that a sign error in
        them is invisible. A test that compares the two over hundreds of random
        states answers that better than a careful read ever would.
        """
        assert self._x is not None
        H = np.zeros((4, 9), dtype=np.float64)
        steps = np.concatenate((np.full(3, 1.0), np.full(3, 0.1), np.full(3, 0.1)))

        for i in range(9):
            step = steps[i]
            forward, backward = self._x.copy(), self._x.copy()
            forward[i] += step
            backward[i] -= step
            H[:, i] = (
                self._measurement(forward, missile) - self._measurement(backward, missile)
            ) / (2.0 * step)
        return H

    def _jacobian(self, missile: EntityState) -> Vector:
        """Linearise the measurement model, analytically.

        With ``dp = p_target - p_missile``, ``dv`` likewise, ``r = |dp|``,
        ``u = dp / r`` and ``b = M dp`` the sightline in the body frame (``M``
        being the body axes, which depend on the *missile's* velocity and so are
        constant with respect to the state being estimated):

        * ``d(range)/d(p) = u``, and range does not depend on velocity at all.
        * ``azimuth = atan2(b1, b0)``, so ``d(az)/db = [-b1, b0, 0] / (b0^2 +
          b1^2)`` and the chain rule through ``b = M dp`` is a multiplication by
          ``M``.
        * ``elevation = asin(-b2 / r)``, giving ``d(el)/db = [b2 b0, b2 b1,
          b2^2 - r^2] / r^3`` scaled by ``1 / sqrt(1 - sin^2)``.
        * ``range_rate = (dp . dv) / r``, so ``d/d(p) = (dv - range_rate u) / r``
          and ``d/d(v) = u``.

        Nothing depends on the target's acceleration — the seeker measures where
        the target *is* and how fast the range is changing, not how it is
        turning. That column of zeros is exactly why the filter needs a motion
        model to estimate acceleration at all, and why APN depends on the model
        being right rather than on the measurement being good.

        Guarded twice. A target directly above or below the nose makes azimuth
        undefined, and one exactly on the boresight makes elevation's derivative
        infinite. Both are far outside a 40-degree gimbal limit in any real
        engagement, but a filter can pass through anything while it is still
        converging, and a NaN in the Jacobian poisons the covariance for good.
        """
        assert self._x is not None
        H = np.zeros((4, 9), dtype=np.float64)

        relative = self._x[0:3] - missile.pos
        relative_velocity = self._x[3:6] - missile.vel
        distance = magnitude(relative)
        if distance < _EPS:
            return H

        axes = body_axes(missile.vel)
        body = axes @ relative
        unit_sightline = relative / distance

        # Range.
        H[0, 0:3] = unit_sightline

        # Azimuth, in the horizontal plane of the body frame.
        horizontal = float(body[0] ** 2 + body[1] ** 2)
        if horizontal > _EPS:
            H[1, 0:3] = np.array([-body[1], body[0], 0.0]) @ axes / horizontal

        # Elevation.
        sine = float(np.clip(-body[2] / distance, -1.0, 1.0))
        cosine = float(np.sqrt(max(1.0 - sine * sine, _EPS)))
        d_sine = (
            np.array(
                [
                    body[2] * body[0],
                    body[2] * body[1],
                    body[2] * body[2] - distance * distance,
                ]
            )
            / distance**3
        )
        H[2, 0:3] = (d_sine @ axes) / cosine

        # Range rate.
        range_rate = float(np.dot(relative, relative_velocity)) / distance
        H[3, 0:3] = (relative_velocity - range_rate * unit_sightline) / distance
        H[3, 3:6] = unit_sightline

        return H

    def _measurement_noise(self, distance: float) -> Vector:
        """Measurement covariance, which depends on range through glint.

        Angle error has two parts that behave oppositely: the seeker's own noise
        is a fixed angle, while glint is a fixed *distance* and so subtends a
        growing angle as the range falls. Adding them in quadrature gives the
        filter an honest picture of when to trust its angles — and it is why the
        estimate degrades near the end of an engagement rather than converging.
        """
        glint_angle = self.seeker.glint_sigma / max(distance, 1.0)
        angle_variance = self.seeker.angle_sigma**2 + glint_angle**2
        return np.diag(
            np.array(
                [
                    max(self.seeker.range_sigma**2, 1e-6),
                    max(angle_variance, 1e-12),
                    max(angle_variance, 1e-12),
                    max(self.seeker.range_rate_sigma**2, 1e-6),
                ],
                dtype=np.float64,
            )
        )

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        if self._x is None or dt <= 0.0:
            return
        F = self._transition(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)

    def correct(self, measurement: Measurement, missile: EntityState) -> None:
        if self._x is None:
            self._initialise(measurement, missile)
            return

        self._last_time = measurement.time

        predicted = self._measurement(self._x, missile)
        observed = np.array(
            [
                measurement.range,
                measurement.azimuth,
                measurement.elevation,
                measurement.range_rate,
            ],
            dtype=np.float64,
        )
        residual = observed - predicted
        # Angles are cyclic; wrap the residual so a track near +/-pi cannot
        # produce a spurious full-turn correction.
        residual[1] = np.arctan2(np.sin(residual[1]), np.cos(residual[1]))
        residual[2] = np.arctan2(np.sin(residual[2]), np.cos(residual[2]))

        H = self._jacobian(missile)
        R = self._measurement_noise(float(predicted[0]))
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + K @ residual

        # Joseph form. The textbook (I - KH) P is algebraically equivalent but
        # loses symmetry and positive-definiteness to rounding over thousands of
        # cycles; this form stays symmetric by construction.
        identity = np.eye(9, dtype=np.float64)
        A = identity - K @ H
        self._P = A @ self._P @ A.T + K @ R @ K.T

    def _initialise(self, measurement: Measurement, missile: EntityState) -> None:
        """Seed the filter from its first sighting.

        Position comes from the measurement. Velocity and acceleration are
        unknown, so they start at zero with a large variance — which tells the
        filter to believe the next few measurements almost completely, and is
        why the estimate takes about a second to become usable.
        """
        position = missile.pos + relative_position_from(measurement, missile)
        self._x = np.concatenate((position, np.zeros(3), np.zeros(3)))

        self._P = np.eye(9, dtype=np.float64)
        self._P[0:3, 0:3] *= max(self.seeker.range_sigma**2, 100.0)
        self._P[3:6, 3:6] *= self.initial_velocity_sigma**2
        self._P[6:9, 6:9] *= self.initial_acceleration_sigma**2
        self._last_time = measurement.time

    def estimate(self) -> TargetEstimate | None:
        if self._x is None:
            return None
        return TargetEstimate(
            position=self._x[0:3].copy(),
            velocity=self._x[3:6].copy(),
            acceleration=self._x[6:9].copy(),
            covariance=self._P.copy(),
        )

    @property
    def name(self) -> str:
        return f"EKF (jerk sigma={self.jerk_sigma:g})"
