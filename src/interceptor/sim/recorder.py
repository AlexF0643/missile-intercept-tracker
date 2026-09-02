"""State recording into preallocated arrays.

Appending to a Python list inside the physics loop costs more than the physics
does, so the recorder allocates its arrays up front from the known run length
and writes into them by index. It doubles its capacity if a run outlasts the
estimate, which should not happen but should not crash if it does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from interceptor.core.state import Vector

if TYPE_CHECKING:
    from interceptor.core.world import World

__all__ = ["Recorder"]


class Recorder:
    """Time histories of every entity's position, velocity and mass."""

    def __init__(self, names: tuple[str, ...], capacity: int) -> None:
        if capacity < 1:
            msg = f"capacity must be at least 1, got {capacity}"
            raise ValueError(msg)
        self.names = names
        self._count = 0
        self._time: Vector = np.zeros(capacity, dtype=np.float64)
        self._pos: dict[str, Vector] = {n: np.zeros((capacity, 3), dtype=np.float64) for n in names}
        self._vel: dict[str, Vector] = {n: np.zeros((capacity, 3), dtype=np.float64) for n in names}
        self._mass: dict[str, Vector] = {n: np.zeros(capacity, dtype=np.float64) for n in names}
        self._cmd: dict[str, Vector] = {n: np.zeros(capacity, dtype=np.float64) for n in names}
        self._acc: dict[str, Vector] = {n: np.zeros(capacity, dtype=np.float64) for n in names}
        self._lim: dict[str, Vector] = {n: np.zeros(capacity, dtype=np.float64) for n in names}

    def __len__(self) -> int:
        return self._count

    def _grow(self) -> None:
        new_capacity = max(1, self._time.size * 2)
        self._time = np.resize(self._time, new_capacity)
        for name in self.names:
            self._pos[name] = np.resize(self._pos[name], (new_capacity, 3))
            self._vel[name] = np.resize(self._vel[name], (new_capacity, 3))
            self._mass[name] = np.resize(self._mass[name], new_capacity)
            self._cmd[name] = np.resize(self._cmd[name], new_capacity)
            self._acc[name] = np.resize(self._acc[name], new_capacity)
            self._lim[name] = np.resize(self._lim[name], new_capacity)

    def record(self, t: float, world: World) -> None:
        """Append one sample of the whole world."""
        if self._count >= self._time.size:
            self._grow()
        i = self._count
        self._time[i] = t
        for entity in world.entities:
            if entity.name not in self._pos:
                continue
            self._pos[entity.name][i] = entity.state.pos
            self._vel[entity.name][i] = entity.state.vel
            self._mass[entity.name][i] = entity.state.mass
            self._cmd[entity.name][i] = entity.commanded_acceleration()
            self._acc[entity.name][i] = entity.achieved_acceleration()
            self._lim[entity.name][i] = entity.acceleration_limit()
        self._count += 1

    @property
    def time(self) -> Vector:
        """Sample times, seconds."""
        return self._time[: self._count]

    def position(self, name: str) -> Vector:
        """Position history for one entity, shape ``(n, 3)``."""
        return self._pos[name][: self._count]

    def velocity(self, name: str) -> Vector:
        """Velocity history for one entity, shape ``(n, 3)``."""
        return self._vel[name][: self._count]

    def mass(self, name: str) -> Vector:
        """Mass history for one entity, shape ``(n,)``."""
        return self._mass[name][: self._count]

    def speed(self, name: str) -> Vector:
        """Speed history for one entity, shape ``(n,)``."""
        return np.asarray(np.linalg.norm(self.velocity(name), axis=1), dtype=np.float64)

    def commanded(self, name: str) -> Vector:
        """Guidance-commanded acceleration magnitude, m/s^2, shape ``(n,)``."""
        return self._cmd[name][: self._count]

    def achieved(self, name: str) -> Vector:
        """Acceleration actually applied after limiting and lag, m/s^2."""
        return self._acc[name][: self._count]

    def limit(self, name: str) -> Vector:
        """The airframe's available lateral acceleration at each sample, m/s^2."""
        return self._lim[name][: self._count]

    def separation(self, first: str, second: str) -> Vector:
        """Distance between two entities at each sample, metres."""
        delta = self.position(first) - self.position(second)
        return np.asarray(np.linalg.norm(delta, axis=1), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Write the run to a compressed ``.npz`` file."""
        payload: dict[str, Vector] = {"time": self.time}
        for name in self.names:
            payload[f"{name}.pos"] = self.position(name)
            payload[f"{name}.vel"] = self.velocity(name)
            payload[f"{name}.mass"] = self.mass(name)
        # numpy's stubs type **kwds against `allow_pickle: bool` rather than as
        # arrays, so a dict splat is rejected despite being the documented API.
        np.savez_compressed(Path(path), **payload)  # type: ignore[arg-type]
