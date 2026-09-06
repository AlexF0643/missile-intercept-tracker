"""The browser app: one window with the 3D view and every parameter.

Nothing in the simulation imports this, and this imports nothing the simulation
does not already have. It is a renderer and a form, over the standard library.
"""

from __future__ import annotations

from interceptor.web.app import address_of, make_server, serve

__all__ = ["address_of", "make_server", "serve"]
