"""Integrator accuracy and convergence order."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interceptor.core.integrate import Derivative, euler, rk4
from interceptor.core.state import Vector

#: The shape shared by every integrator: ``method(f, t, y, dt) -> y_next``.
Method = Callable[[Derivative, float, Vector, float], Vector]


def _decay(t: float, y: Vector) -> Vector:
    """dy/dt = -2y, whose solution is exp(-2t)."""
    del t
    return -2.0 * y


def _cubic_slope(t: float, y: Vector) -> Vector:
    """dy/dt = 3t^2, whose solution is t^3."""
    del y
    return np.array([3.0 * t * t])


def _negate(t: float, y: Vector) -> Vector:
    del t
    return -y


def _integrate(method: Method, f: Derivative, dt: float, steps: int) -> float:
    y = np.array([1.0])
    for i in range(steps):
        y = method(f, i * dt, y, dt)
    return float(y[0])


def _rk4_decay_error(dt: float) -> float:
    steps = round(1.0 / dt)
    final = _integrate(rk4, _decay, dt, steps)
    return abs(final - float(np.exp(-2.0)))


def _euler_decay_error(dt: float) -> float:
    steps = round(1.0 / dt)
    final = _integrate(euler, _decay, dt, steps)
    return abs(final - float(np.exp(-2.0)))


def test_rk4_is_accurate_on_exponential_decay() -> None:
    assert _rk4_decay_error(0.01) < 1e-9


def test_rk4_converges_at_fourth_order() -> None:
    """Halving the step should cut the error by roughly 2^4."""
    coarse = _rk4_decay_error(0.1)
    fine = _rk4_decay_error(0.05)
    assert 10.0 < coarse / fine < 22.0


def test_euler_converges_at_first_order() -> None:
    """Halving the step should cut the error by roughly 2 — the contrast matters."""
    coarse = _euler_decay_error(0.1)
    fine = _euler_decay_error(0.05)
    assert 1.7 < coarse / fine < 2.3


def test_euler_is_far_worse_than_rk4_at_the_same_step() -> None:
    assert _euler_decay_error(0.01) > 1000.0 * _rk4_decay_error(0.01)


def test_rk4_is_exact_on_a_cubic() -> None:
    """RK4 integrates polynomials up to degree four without truncation error.

    This is why the ballistic test lands at the floating-point floor rather than
    merely close: a constant-acceleration trajectory is a quadratic.
    """
    y = np.array([0.0])
    dt = 0.01
    for i in range(100):
        y = rk4(_cubic_slope, i * dt, y, dt)
    assert abs(float(y[0]) - 1.0) < 1e-12


def test_both_integrators_leave_their_input_unmodified() -> None:
    y = np.array([1.0, 2.0, 3.0])
    original = y.copy()
    rk4(_negate, 0.0, y, 0.01)
    euler(_negate, 0.0, y, 0.01)
    assert np.array_equal(y, original)
