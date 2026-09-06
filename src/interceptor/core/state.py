"""Entity state, and its flat-array representation for the integrator.

Every position and velocity in this project is a length-3 NumPy array in the
world frame. Nothing anywhere stores an angle-and-magnitude pair, because that
is how a codebase silently becomes two-dimensional.

The world frame is ENU: x east, y north, z up, in metres. Altitude is therefore
just ``pos[2]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["STATE_SIZE", "EntityState", "Vector", "magnitude"]

#: An array of 64-bit floats — length 3 for every position, velocity and
#: acceleration in the package. Spelled as an explicit ``TypeAlias`` so that a
#: type checker reads it as a type rather than as a module-level variable.
Vector: TypeAlias = NDArray[np.float64]

#: Width of the flat state vector: three position, three velocity, one mass.
STATE_SIZE: Final = 7


def magnitude(v: Vector) -> float:
    """Length of a three-vector.

    ``float(np.linalg.norm(v))`` with the generality removed. ``norm`` supports
    matrix norms, arbitrary orders, axes and keepdims, and dispatches through
    all of that on every call; for the 2-norm of a 1-D array it ends up doing
    ``sqrt(dot(v, v))`` anyway, so this is the same arithmetic and the same
    answer to the last bit — verified in the tests rather than assumed.

    It earns its place by call count. Profiling one engagement found 273,000
    norms, more than any other single operation, because every substep of every
    RK4 step takes a speed, a range and a drag magnitude. At a third of a
    microsecond apiece that is a tenth of the runtime of a Monte Carlo.

    Safe here for the reason it is not safe in general: ``v @ v`` overflows for
    magnitudes above about 1e154, and this simulation works in metres and metres
    per second over a flat earth.
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@dataclass(frozen=True)
class EntityState:
    """Position, velocity and mass of a body at one instant.

    Frozen so that a state cannot be mutated in place behind the integrator's
    back — every step produces a new one. ``__post_init__`` normalises the
    inputs to float64 arrays of shape (3,), so every consumer downstream can
    rely on the dtype and the shape without checking, and a wrong-length vector
    fails at construction rather than somewhere deep in the integrator.
    """

    pos: Vector
    vel: Vector
    mass: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=np.float64).reshape(3))
        object.__setattr__(self, "vel", np.asarray(self.vel, dtype=np.float64).reshape(3))
        object.__setattr__(self, "mass", float(self.mass))
        if self.mass <= 0.0:
            msg = f"mass must be positive, got {self.mass}"
            raise ValueError(msg)

    @property
    def speed(self) -> float:
        """Magnitude of the velocity, in m/s."""
        return magnitude(self.vel)

    @property
    def altitude(self) -> float:
        """Height above the (flat) ground plane, in metres."""
        return float(self.pos[2])

    def to_array(self) -> Vector:
        """Pack into the flat ``[px, py, pz, vx, vy, vz, m]`` the integrator uses."""
        out = np.empty(STATE_SIZE, dtype=np.float64)
        out[0:3] = self.pos
        out[3:6] = self.vel
        out[6] = self.mass
        return out

    @classmethod
    def from_array(cls, y: Vector) -> EntityState:
        """Unpack a flat state vector produced by :meth:`to_array`."""
        if y.shape != (STATE_SIZE,):
            msg = f"expected shape ({STATE_SIZE},), got {y.shape}"
            raise ValueError(msg)
        return cls(pos=y[0:3].copy(), vel=y[3:6].copy(), mass=float(y[6]))
