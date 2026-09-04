"""What to draw, independent of what draws it.

Phase 6 has two renderers: a live VPython window you can orbit around while the
engagement plays, and an offline recorder that writes an animated GIF. Writing
the geometry twice would guarantee they drift apart, and only one of them can be
checked by a test — a live 3D window cannot be asserted about on a build server.

So the geometry lives here, in plain numpy, and both renderers consume it. This
module knows about positions, trails, sightlines and the numbers on the readout;
it knows nothing about cameras, colours or file formats. That makes the part
with the actual arithmetic in it testable, and leaves each renderer as a thin
layer that only has to place objects where it is told.

The other reason for the split is resampling. A run is recorded at 200 Hz and
plays back at 30 frames a second, so something has to decide which samples
become frames — and getting that wrong is how an animation ends up running at
the wrong speed or skipping the intercept, which is the one moment anybody
watches for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from interceptor.core.state import Vector
from interceptor.sim.engagement import RunResult
from interceptor.sim.intercept import Intercept

__all__ = ["Frame", "Storyboard", "storyboard_from"]

_EPS: Final = 1e-12
GRAVITY: Final = 9.80665


@dataclass(frozen=True)
class Frame:
    """One instant, with everything either renderer needs to draw it.

    Trails are cumulative slices rather than growing lists, so a renderer can
    draw the whole path so far without keeping state of its own. They are views
    into the recorded arrays, not copies — a five-minute engagement at 30 fps is
    nine thousand frames, and copying two trajectories into each of them would
    turn a few megabytes into a few hundred.
    """

    time: float
    missile_position: Vector
    target_position: Vector
    missile_trail: Vector
    target_trail: Vector
    range: float
    closing_speed: float
    los_rate: float
    commanded_g: float
    achieved_g: float
    speed: float


@dataclass(frozen=True)
class Storyboard:
    """A whole engagement, resampled to a frame rate.

    Attributes:
        frames: In order, one per rendered frame.
        fps: Frames per second of *simulated* time. Playing them back at this
            rate shows the engagement at life speed.
        bounds: ``(lower, upper)`` corners of a box containing both
            trajectories, so a renderer can set up its axes once rather than
            rescaling every frame — which reads as the world lurching about.
        intercept: The closest approach, if there was one.
        title: What the engagement was.
    """

    frames: list[Frame]
    fps: float
    bounds: tuple[Vector, Vector]
    intercept: Intercept | None
    title: str

    @property
    def duration(self) -> float:
        """Simulated seconds covered."""
        return 0.0 if not self.frames else self.frames[-1].time - self.frames[0].time

    def at(self, when: float) -> Frame:
        """The frame nearest a given simulated time."""
        if not self.frames:
            msg = "storyboard has no frames"
            raise ValueError(msg)
        times = np.array([frame.time for frame in self.frames])
        return self.frames[int(np.argmin(np.abs(times - when)))]


def _los_rate(relative: Vector, relative_velocity: Vector) -> float:
    """Magnitude of the line-of-sight rotation vector.

    Recomputed from the recorded truth rather than taken from the guidance law,
    because what a viewer should show is what the geometry actually did, not
    what the missile believed about it. The two differ once a seeker is
    involved, and the difference is the interesting part.
    """
    denominator = float(np.dot(relative, relative))
    if denominator < _EPS:
        return 0.0
    return float(np.linalg.norm(np.cross(relative, relative_velocity)) / denominator)


def storyboard_from(
    result: RunResult,
    *,
    fps: float = 30.0,
    missile: str = "missile",
    target: str = "target",
    intercept: Intercept | None = None,
    title: str = "",
    speed: float = 1.0,
    slow_motion_from: float | None = None,
    slow_factor: float = 0.25,
) -> Storyboard:
    """Turn a flown engagement into frames.

    Args:
        result: The run to visualise.
        fps: Rendered frames per second of simulated time.
        speed: Playback rate. 1.0 is life speed.
        slow_motion_from: Simulated time after which to slow down. The endgame
            is the part worth watching and the part that passes fastest: at
            Mach 2 the last hundred metres take under a tenth of a second, which
            at 30 fps is three frames of the only thing anybody is looking for.
            ``None`` plays the whole run at one rate.
        slow_factor: How much to slow down after that point. 0.25 is quarter
            speed.
        intercept: Closest approach, drawn as a marker if given.

    Resampling is nearest-neighbour onto an evenly spaced time grid. Not
    interpolation: the recorded samples are the simulation's own truth, and
    inventing intermediate states for the sake of smoothness would mean the
    animation showed positions the simulation never computed. At 200 Hz
    recorded against 30 fps rendered there are six samples per frame, so
    nearest-neighbour is never more than 2.5 ms adrift.
    """
    if fps <= 0.0:
        msg = f"fps must be positive, got {fps}"
        raise ValueError(msg)
    if speed <= 0.0:
        msg = f"speed must be positive, got {speed}"
        raise ValueError(msg)

    recorder = result.recorder
    times = recorder.time
    if len(times) == 0:
        msg = "nothing was recorded, so there is nothing to draw"
        raise ValueError(msg)

    missile_position = recorder.position(missile)
    target_position = recorder.position(target)
    missile_velocity = recorder.velocity(missile)
    target_velocity = recorder.velocity(target)
    commanded = recorder.commanded(missile)
    achieved = recorder.achieved(missile)
    missile_speed = recorder.speed(missile)

    # Frames land on a grid in simulated time. Slowing the playback puts more
    # frames in the same interval rather than showing the same frames longer,
    # which is the difference between slow motion and a stutter.
    start, finish = float(times[0]), float(times[-1])
    step = speed / fps
    if slow_motion_from is None:
        grid = np.arange(start, finish + 0.5 * step, step)
    else:
        if not 0.0 < slow_factor <= 1.0:
            msg = f"slow_factor must be in (0, 1], got {slow_factor}"
            raise ValueError(msg)
        boundary = float(np.clip(slow_motion_from, start, finish))
        slow_step = step * slow_factor
        grid = np.concatenate(
            (
                np.arange(start, boundary, step),
                np.arange(boundary, finish + 0.5 * slow_step, slow_step),
            )
        )
    indices = np.searchsorted(times, grid).clip(0, len(times) - 1)

    frames = [
        Frame(
            time=float(times[i]),
            missile_position=missile_position[i],
            target_position=target_position[i],
            missile_trail=missile_position[: i + 1],
            target_trail=target_position[: i + 1],
            range=float(np.linalg.norm(target_position[i] - missile_position[i])),
            closing_speed=_closing(
                target_position[i] - missile_position[i],
                target_velocity[i] - missile_velocity[i],
            ),
            los_rate=_los_rate(
                target_position[i] - missile_position[i],
                target_velocity[i] - missile_velocity[i],
            ),
            # The recorder stores these as magnitudes already, shape (n,).
            commanded_g=float(commanded[i]) / GRAVITY,
            achieved_g=float(achieved[i]) / GRAVITY,
            speed=float(missile_speed[i]),
        )
        for i in indices
    ]

    together = np.vstack((missile_position, target_position))
    bounds = (together.min(axis=0), together.max(axis=0))

    return Storyboard(frames=frames, fps=fps, bounds=bounds, intercept=intercept, title=title)


def _closing(relative: Vector, relative_velocity: Vector) -> float:
    """Rate at which range is shrinking, positive while closing."""
    distance = float(np.linalg.norm(relative))
    if distance < _EPS:
        return 0.0
    return -float(np.dot(relative, relative_velocity)) / distance
