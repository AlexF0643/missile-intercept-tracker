"""Proportional navigation.

The idea in one sentence: if the bearing to a target is not changing while the
range closes, you are on a collision course — so turn in proportion to how fast
the bearing is drifting, until it stops drifting.

That is the whole law. It steers at where the target is *going* rather than
where it is, and it does so without ever computing an intercept point, without
predicting the target's future path, and without knowing anything about the
target's speed or heading. It needs one quantity: the rate at which the
sightline is rotating.

    a = N * V_c * (Omega cross r_hat)

where ``Omega = (r cross v) / (r . r)`` is the line-of-sight rotation vector,
``V_c`` is the closing velocity and ``r_hat`` is the unit sightline. Since
``Omega`` is perpendicular to ``r_hat`` by construction, the cross product has
magnitude ``|Omega|``, so the command's magnitude is ``N * V_c * lambda_dot`` —
the classical scalar law, lifted into three dimensions with no angles and no
special cases.

``N = 3`` is not arbitrary: it is the optimal navigation constant against a
non-manoeuvring target under a minimum-control-effort criterion. Values between
3 and 5 are the practical range. Beyond about 5 the loop amplifies seeker noise
into wasted lateral acceleration faster than it improves the intercept — an
effect that is invisible on truth data and becomes the dominant one in Phase 4.
"""

from __future__ import annotations

import numpy as np

from interceptor.core.state import EntityState, Vector
from interceptor.guidance.base import GuidanceLaw, perpendicular_component
from interceptor.sensing.track import Track

__all__ = ["AugmentedProportionalNavigation", "ProportionalNavigation"]

_EPS = 1e-9


class ProportionalNavigation(GuidanceLaw):
    """Classical proportional navigation, in vector form.

    Args:
        navigation_constant: ``N``. Defaults to 3, the optimal value against a
            non-manoeuvring target.

    The command is projected perpendicular to the missile's velocity before it
    is returned. That projection is what makes this *pure* proportional
    navigation rather than the *true* variant: an aerodynamic missile generates
    force by lift, and lift acts at right angles to the airflow, so a component
    along the velocity is not something the airframe could produce. The two
    forms converge as the lead angle goes to zero, which is where a healthy
    engagement spends its last seconds anyway.
    """

    def __init__(self, navigation_constant: float = 3.0) -> None:
        if navigation_constant <= 0.0:
            msg = f"navigation_constant must be positive, got {navigation_constant}"
            raise ValueError(msg)
        self.navigation_constant = navigation_constant

    @property
    def name(self) -> str:
        return f"ProNav (N={self.navigation_constant:g})"

    def command(self, track: Track, missile: EntityState) -> Vector:
        if not track.valid:
            return np.zeros(3, dtype=np.float64)

        closing = track.closing_speed
        if closing <= _EPS:
            # Not closing. PN is undefined here — its whole premise is that the
            # range is shrinking — and commanding anything would be a guess.
            return np.zeros(3, dtype=np.float64)

        # |Omega x r_hat| = |Omega| because Omega is perpendicular to r_hat by
        # construction, so this has magnitude N * V_c * lambda_dot exactly.
        command = (
            self.navigation_constant
            * closing
            * np.cross(track.los_rate_vector, track.line_of_sight)
        )

        if missile.speed < _EPS:
            return np.asarray(command, dtype=np.float64)
        return perpendicular_component(command, missile.vel)


class AugmentedProportionalNavigation(ProportionalNavigation):
    """Proportional navigation with a term for the target's own acceleration.

        a = N * V_c * (Omega x r_hat)  +  (N / 2) * a_t_perp

    Plain PN assumes the target holds its velocity. Everything it does follows
    from that assumption, and against a target that manoeuvres it is wrong in a
    specific, correctable way: the sightline is being driven by an acceleration
    the law does not know about, so PN spends the engagement reacting to a drift
    it could have anticipated. It always arrives a little late, and "a little
    late" is the whole miss distance in the last second.

    The augmented term supplies what was missing. The target's acceleration is
    projected perpendicular to the sightline — the component along it changes
    the closing speed rather than the bearing, and PN steers on bearing — and
    added with a coefficient of ``N / 2``. That coefficient is not a tuning
    knob: it is the optimal-control solution for a constant-acceleration target
    under the same minimum-effort criterion that gives ``N = 3`` for a
    non-manoeuvring one.

    **What it costs.** APN needs an estimate of target acceleration, and that is
    the hardest thing in the whole sensing chain to measure — it is the second
    derivative of a noisy position. Feed it a bad estimate and the augmented
    term injects noise straight into the command at ``N / 2`` times its
    magnitude, and that is not hypothetical: against a jinking target the EKF's
    acceleration error over the last two seconds runs at a median 12.0 g while
    the target pulls 7.0, so the lead contributes some 18 g of noise and APN
    ends up worse than the law it augments. Against a weave the same filter is
    wrong by 1.8 g and the same term is worth a factor of six. That is why this
    arrives only now: the extended Kalman filter
    produces the estimate, and the NEES check in
    :mod:`interceptor.sensing.consistency` established that its stated
    uncertainty is honest. Without both, the term is a liability.

    **Where it wins, measured.** Through the seeker and the EKF, six seeds on the
    crossing engagement: the 6 g weave goes from 6.94 m and no hits at all to
    1.21 m and six from six; the 7 g break turn from 4.87 m and three from six to
    1.62 m and six from six. Both targets hold their acceleration long enough for
    the assumption to be true, and where it is true this term is worth roughly a
    factor of five.

    **Where it loses, and why that is a property rather than a bug.** Against a
    barrel roll it misses by 469 m where plain PN misses by 31 — and it does so
    on a *perfect* track, which rules the filter out of the diagnosis. A barrel
    roll holds its acceleration *magnitude* constant while rotating its
    *direction*, so the lead never decays: the missile carries a standing 7.5 g
    command pointing somewhere different every second. It commands less peak
    acceleration than PN does (13 g against 246 g) and yet uses more on average,
    pays induced drag for every bit of it, and arrives at 265 m/s where PN
    arrives at 407, with nothing left to correct with. The assumption here is not
    merely unhelpful, it is expensive.

    This is left as it is, and asserted in ``tests/test_apn.py`` rather than
    fixed. Switching laws on whichever currently wins would hide the one thing
    worth knowing about this one: exactly which assumption it rests on, and what
    happens when the world declines to satisfy it.

    Degrades to plain PN when the track carries no acceleration — an alpha-beta
    filter reports zero, a bare seeker track reports ``None`` — and does so
    silently, because a guidance law asking its track for something it cannot
    supply should get on with what it can.
    """

    @property
    def name(self) -> str:
        return f"APN (N={self.navigation_constant:g})"

    def command(self, track: Track, missile: EntityState) -> Vector:
        base = super().command(track, missile)

        acceleration = track.target_acceleration
        if acceleration is None or not track.valid:
            return base
        if track.closing_speed <= _EPS:
            # PN returned zero here for the same reason: not closing, nothing
            # sensible to command. Adding a lead term to that would be worse
            # than nothing.
            return base

        # Only the component across the sightline matters. Acceleration along it
        # changes how fast the range closes, which is not what this law steers on.
        sightline = track.line_of_sight
        perpendicular = acceleration - float(np.dot(acceleration, sightline)) * sightline
        lead = 0.5 * self.navigation_constant * perpendicular

        if missile.speed < _EPS:
            return np.asarray(base + lead, dtype=np.float64)
        return perpendicular_component(base + lead, missile.vel)
