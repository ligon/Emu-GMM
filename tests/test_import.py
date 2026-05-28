"""Smoke test: the package imports and exposes a version string."""

import emu_gmm


def test_import():
    assert hasattr(emu_gmm, "__version__")
    assert isinstance(emu_gmm.__version__, str)
