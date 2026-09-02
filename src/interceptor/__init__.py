"""A 3-DOF missile intercept simulation.

The package is organised in layers that can be tested independently:

``core``
    Truth: entity state, the fixed-step integrator, frame conversions and the
    world that steps everything forward.
``airframe``
    Propulsion, aerodynamics and the autopilot that limits what the missile can
    physically be asked to do.
``entities``
    The bodies that occupy the world: the missile and its target.
``sensing``
    The seeker and the estimator — the missile's lossy window onto the truth.
``guidance``
    Laws that turn an estimated track into a commanded lateral acceleration.
``sim``
    The multi-rate loop, state recording and the engagement runner.
``viz``
    Rendering. Nothing else in the package may import from here, so that
    headless machines and CI can run everything except the viewer.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
