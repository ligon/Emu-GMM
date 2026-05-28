"""emu-gmm: Measure-theoretic GMM. Estimation via E_mu.

See docs/design.org for the architectural specification, docs/api-sketch.org
for the v1 API surface, and docs/mcar-asymptotics.org for the asymptotic
theory under MCAR.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("emu-gmm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
