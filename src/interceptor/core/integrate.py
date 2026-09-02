"""Fixed-step numerical integrators.

Pure functions over flat state vectors: no objects, no hidden state, no
adaptive stepping. Reproducibility depends on the step being fixed — an
adaptive integrator would make two runs of the same scenario differ in their
sampling, and every downstream comparison meaningless.
"""

from __future__ import annotations

from collections.abc import Callable

from interceptor.core.state import Vector

__all__ = ["Derivative", "euler", "rk4"]

#: A function ``f(t, y)`` returning ``dy/dt`` for the flat state ``y``.
Derivative = Callable[[float, Vector], Vector]


def euler(f: Derivative, t: float, y: Vector, dt: float) -> Vector:
    """Advance one step by explicit Euler. First-order accurate.

    Present for comparison in the tests, not for use in the simulation: it
    accumulates error fast enough to be visible in a ten-second flight.
    """
    return y + dt * f(t, y)


def rk4(f: Derivative, t: float, y: Vector, dt: float) -> Vector:
    """Advance one step by classical fourth-order Runge-Kutta.

    Four derivative evaluations per step. Halving ``dt`` cuts the local error by
    roughly a factor of sixteen, and a system whose exact solution is a
    polynomial of degree four or less is integrated exactly up to floating-point
    rounding — which is why a drag-free ballistic trajectory comes out at the
    limit of double precision rather than merely close.
    """
    half = 0.5 * dt
    k1 = f(t, y)
    k2 = f(t + half, y + half * k1)
    k3 = f(t + half, y + half * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
