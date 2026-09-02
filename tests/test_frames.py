"""Frame conversions.

The round-trip tests here are the highest-value tests in Phase 1. A sign error
or a swapped axis in a rotation does not crash and does not look wrong — it
produces a perfectly smooth trajectory that is simply incorrect, and it will be
blamed on the guidance law for a week. Converting a thousand random vectors out
and back and demanding the original returns rules that out mechanically.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.core.frames import (
    az_el_from_enu,
    az_el_from_frd,
    body_axes,
    body_to_world,
    enu_from_az_el,
    frd_from_az_el,
    unit,
    world_to_body,
)

N_RANDOM = 1000
SEED = 20260902


def _random_vectors(count: int = N_RANDOM, scale: float = 5000.0) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.uniform(-scale, scale, size=(count, 3))


def test_enu_spherical_round_trip_is_exact() -> None:
    worst = 0.0
    for v in _random_vectors():
        rng_, azimuth, elevation = az_el_from_enu(v)
        worst = max(worst, float(np.abs(enu_from_az_el(rng_, azimuth, elevation) - v).max()))
    assert worst < 1e-9, f"worst round-trip error {worst:.3e} m"


def test_frd_spherical_round_trip_is_exact() -> None:
    worst = 0.0
    for v in _random_vectors():
        rng_, azimuth, elevation = az_el_from_frd(v)
        worst = max(worst, float(np.abs(frd_from_az_el(rng_, azimuth, elevation) - v).max()))
    assert worst < 1e-9, f"worst round-trip error {worst:.3e} m"


def test_enu_azimuth_conventions() -> None:
    """North is zero, east is +90 degrees, and elevation is positive upward."""
    _, azimuth, elevation = az_el_from_enu(np.array([0.0, 1.0, 0.0]))
    assert azimuth == pytest.approx(0.0)
    assert elevation == pytest.approx(0.0)

    _, azimuth, _ = az_el_from_enu(np.array([1.0, 0.0, 0.0]))
    assert azimuth == pytest.approx(np.pi / 2)

    _, _, elevation = az_el_from_enu(np.array([0.0, 0.0, 1.0]))
    assert elevation == pytest.approx(np.pi / 2)


def test_frd_angle_conventions() -> None:
    """Straight ahead is zero; right is +azimuth; up is +elevation."""
    _, azimuth, elevation = az_el_from_frd(np.array([1.0, 0.0, 0.0]))
    assert azimuth == pytest.approx(0.0)
    assert elevation == pytest.approx(0.0)

    _, azimuth, _ = az_el_from_frd(np.array([0.0, 1.0, 0.0]))
    assert azimuth == pytest.approx(np.pi / 2)

    _, _, elevation = az_el_from_frd(np.array([0.0, 0.0, -1.0]))
    assert elevation == pytest.approx(np.pi / 2)


def test_body_axes_are_orthonormal_and_right_handed() -> None:
    for v in _random_vectors(scale=800.0):
        axes = body_axes(v)
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(axes)) == pytest.approx(1.0, abs=1e-12)


def test_body_forward_axis_is_the_direction_of_travel() -> None:
    for v in _random_vectors(scale=800.0):
        assert np.allclose(body_axes(v)[0], unit(v), atol=1e-12)


def test_body_right_axis_is_horizontal_in_level_flight() -> None:
    """Roll is resolved wings-level, so the right axis has no vertical component."""
    axes = body_axes(np.array([120.0, 250.0, 30.0]))
    assert axes[1][2] == pytest.approx(0.0, abs=1e-12)


def test_world_body_round_trip_is_exact() -> None:
    axes = body_axes(np.array([300.0, -120.0, 45.0]))
    worst = 0.0
    for v in _random_vectors(count=200):
        recovered = body_to_world(axes, world_to_body(axes, v))
        worst = max(worst, float(np.abs(recovered - v).max()))
    assert worst < 1e-9


def test_vertical_flight_does_not_degenerate() -> None:
    """Straight up is the degenerate case for a wings-level roll convention."""
    for direction in ([0.0, 0.0, 400.0], [0.0, 0.0, -400.0]):
        axes = body_axes(np.array(direction))
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(axes)) == pytest.approx(1.0, abs=1e-12)


def test_unit_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        unit(np.zeros(3))


def test_body_axes_rejects_a_body_at_rest() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        body_axes(np.zeros(3))
