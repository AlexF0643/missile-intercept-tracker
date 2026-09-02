"""Phase 0 smoke test.

There is no simulation yet. This asserts only that the package is installable,
importable and reports a version — which is enough to prove the packaging, the
src layout and the CI workflow all work before any physics depends on them.
"""

from __future__ import annotations

import interceptor


def test_package_imports() -> None:
    assert interceptor.__doc__ is not None


def test_version_is_semver_shaped() -> None:
    parts = interceptor.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
