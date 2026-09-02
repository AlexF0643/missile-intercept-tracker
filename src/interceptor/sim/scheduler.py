"""Multi-rate tick scheduling.

Every subsystem rate is derived from the one physics tick, as an exact integer
divisor. Deriving rates rather than tracking separate clocks means nothing can
drift apart over a long run, and it makes "the guidance ran 412 times" an exact
statement rather than an approximate one.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Rate", "Scheduler"]


@dataclass(frozen=True)
class Rate:
    """A subsystem rate, expressed as a whole number of physics ticks."""

    hz: float
    interval_ticks: int

    def due(self, tick: int) -> bool:
        """True on the ticks this rate fires."""
        return tick % self.interval_ticks == 0


class Scheduler:
    """Derives subsystem rates from the base physics rate."""

    def __init__(self, base_hz: float) -> None:
        if base_hz <= 0.0:
            msg = f"base_hz must be positive, got {base_hz}"
            raise ValueError(msg)
        self.base_hz = base_hz

    @property
    def dt(self) -> float:
        """The physics timestep, seconds."""
        return 1.0 / self.base_hz

    def rate(self, hz: float) -> Rate:
        """Build a :class:`Rate` for ``hz``.

        Raises:
            ValueError: if ``hz`` does not divide the base rate exactly. A
                seeker at 100 Hz under a 1 kHz physics step is 10 ticks; a
                seeker at 30 Hz would be 33.33, and silently rounding it would
                make the simulation's timing a lie.
        """
        if hz <= 0.0:
            msg = f"rate must be positive, got {hz}"
            raise ValueError(msg)
        exact = self.base_hz / hz
        interval = round(exact)
        if interval < 1 or abs(exact - interval) > 1e-9:
            msg = (
                f"{hz} Hz does not divide the {self.base_hz} Hz physics rate exactly "
                f"({exact:.6f} ticks); choose a rate that does"
            )
            raise ValueError(msg)
        return Rate(hz=hz, interval_ticks=interval)
