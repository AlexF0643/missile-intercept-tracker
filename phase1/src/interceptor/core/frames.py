"""Coordinate frames and the conversions between them.

Two frames are in play.

**World (ENU)** — x east, y north, z up. Everything the simulation stores as
truth lives here. Azimuth is measured from north towards east, the way a
compass bearing runs; elevation is measured up from the horizontal plane.

**Body (FRD)** — x forward, y right, z down, the standard aerospace body triad.
In this 3-DOF model the missile has no independent attitude, so the body frame
is derived from the velocity vector: forward *is* the direction of travel. That
is the zero-angle-of-attack assumption, and it is the single largest fidelity
limit in the whole model. Off-boresight angles for the seeker (Phase 4) are
measured in this frame: azimuth positive to the right, elevation positive up.

A round-trip through either conversion is exact to floating point, which
:mod:`tests.test_frames` asserts over a thousand random vectors. That one test
rules out an entire category of sign and convention bug that is otherwise
almost impossible to spot — a wrong rotation produces a trajectory that looks
perfectly plausible and is simply incorrect.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from interceptor.core.state import Vector

__all__ = [
    "WORLD_NORTH",
    "WORLD_UP",
    "az_el_from_enu",
    "az_el_from_frd",
    "body_axes",
    "body_to_world",
    "enu_from_az_el",
    "frd_from_az_el",
    "unit",
    "world_to_body",
]

WORLD_UP: Final[Vector] = np.array([0.0, 0.0, 1.0])
WORLD_NORTH: Final[Vector] = np.array([0.0, 1.0, 0.0])

# Below this length a vector has no meaningful direction.
_EPS: Final = 1e-12


def unit(v: Vector) -> Vector:
    """Return ``v`` scaled to unit length.

    Raises:
        ValueError: if ``v`` is too short to have a direction. Failing loudly
            here is deliberate — silently returning a zero vector would produce
            a missile that coasts in an arbitrary direction with no error.
    """
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        msg = "cannot take the direction of a zero-length vector"
        raise ValueError(msg)
    return np.asarray(v / norm, dtype=np.float64)


def az_el_from_enu(v: Vector) -> tuple[float, float, float]:
    """Convert a world-frame vector to ``(range, azimuth, elevation)``.

    Azimuth is radians clockwise from north; elevation is radians above the
    horizontal. Range is in whatever unit ``v`` was.
    """
    rng = float(np.linalg.norm(v))
    if rng < _EPS:
        return 0.0, 0.0, 0.0
    azimuth = float(np.arctan2(v[0], v[1]))
    elevation = float(np.arcsin(np.clip(v[2] / rng, -1.0, 1.0)))
    return rng, azimuth, elevation


def enu_from_az_el(rng: float, azimuth: float, elevation: float) -> Vector:
    """Inverse of :func:`az_el_from_enu`."""
    horizontal = rng * np.cos(elevation)
    return np.array(
        [
            horizontal * np.sin(azimuth),
            horizontal * np.cos(azimuth),
            rng * np.sin(elevation),
        ],
        dtype=np.float64,
    )


def az_el_from_frd(v: Vector) -> tuple[float, float, float]:
    """Convert a body-frame vector to ``(range, azimuth, elevation)``.

    Azimuth is radians to the right of the nose; elevation is radians above the
    nose. Both are zero for a target directly on the boresight.
    """
    rng = float(np.linalg.norm(v))
    if rng < _EPS:
        return 0.0, 0.0, 0.0
    azimuth = float(np.arctan2(v[1], v[0]))
    elevation = float(np.arcsin(np.clip(-v[2] / rng, -1.0, 1.0)))
    return rng, azimuth, elevation


def frd_from_az_el(rng: float, azimuth: float, elevation: float) -> Vector:
    """Inverse of :func:`az_el_from_frd`."""
    forward = rng * np.cos(elevation)
    return np.array(
        [
            forward * np.cos(azimuth),
            forward * np.sin(azimuth),
            -rng * np.sin(elevation),
        ],
        dtype=np.float64,
    )


def body_axes(velocity: Vector) -> Vector:
    """Build the body-frame rotation matrix from a velocity vector.

    Returns a 3x3 array whose rows are the forward, right and down unit vectors
    expressed in world coordinates. It is orthonormal with determinant +1, and
    multiplying a world vector by it gives that vector in the body frame.

    Roll is not a free parameter in 3 DOF, so it is resolved by keeping the
    right-hand axis horizontal — the aerodynamic "wings level" convention. When
    the body is flying straight up or down that is degenerate, and north is used
    as the reference direction instead; the resulting roll is arbitrary but
    consistent, which is all any downstream calculation needs.

    Raises:
        ValueError: if ``velocity`` has no direction.
    """
    forward = unit(velocity)

    reference = WORLD_UP
    if abs(float(np.dot(forward, WORLD_UP))) > 1.0 - 1e-9:
        reference = WORLD_NORTH

    right = unit(np.cross(forward, reference))
    down = np.cross(forward, right)
    return np.array([forward, right, down], dtype=np.float64)


def world_to_body(axes: Vector, v: Vector) -> Vector:
    """Express a world-frame vector in the body frame given ``body_axes``."""
    return np.asarray(axes @ v, dtype=np.float64)


def body_to_world(axes: Vector, v: Vector) -> Vector:
    """Express a body-frame vector in the world frame given ``body_axes``.

    The inverse of an orthonormal matrix is its transpose, so this is exact
    rather than a solve.
    """
    return np.asarray(axes.T @ v, dtype=np.float64)
