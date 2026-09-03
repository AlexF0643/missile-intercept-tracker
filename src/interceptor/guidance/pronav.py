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

__all__ = ["ProportionalNavigation"]

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
