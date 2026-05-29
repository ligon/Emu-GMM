"""emu-gmm: Measure-theoretic GMM. Estimation via E_mu.

The public API is re-exported here for convenience::

    from emu_gmm import (
        estimate,
        SyntheticMeasure, SyntheticCovariance,
        AnalyticalMeasure, AnalyticalCovariance,
        EmpiricalMeasure, IIDCovariance, ClusteredCovariance,
        Identity, Fixed, ContinuouslyUpdated,
        DiagonalTikhonov,
        optimistix_lm, scipy_lm,
        EstimationResult, Diagnostics, OptimizerInfo,
    )

See docs/design.org for the architectural specification, docs/api-sketch.org
for the v1 API surface, and docs/mcar-asymptotics.org for the asymptotic
theory under MCAR.
"""

from importlib.metadata import PackageNotFoundError, version

# Public re-exports.
from emu_gmm.covariance import (
    AnalyticalCovariance,
    ClusteredCovariance,
    IIDCovariance,
    SyntheticCovariance,
)
from emu_gmm.estimator import estimate
from emu_gmm.measures import (
    AnalyticalMeasure,
    EmpiricalMeasure,
    SyntheticMeasure,
)
from emu_gmm.optimizer import optimistix_lm, scipy_lm
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    Diagnostics,
    EstimationResult,
    Measure,
    Optimizer,
    OptimizerInfo,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import ContinuouslyUpdated, Fixed, Identity

try:
    __version__ = version("emu-gmm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Entry point
    "estimate",
    # Measures
    "SyntheticMeasure",
    "AnalyticalMeasure",
    "EmpiricalMeasure",
    # Covariance strategies
    "SyntheticCovariance",
    "AnalyticalCovariance",
    "IIDCovariance",
    "ClusteredCovariance",
    # Weighting strategies
    "Identity",
    "Fixed",
    "ContinuouslyUpdated",
    # Regularization
    "DiagonalTikhonov",
    # Optimisers
    "optimistix_lm",
    "scipy_lm",
    # Result / diagnostics types
    "EstimationResult",
    "Diagnostics",
    "OptimizerInfo",
    # Protocols (for type-checking user code)
    "Measure",
    "CovarianceStrategy",
    "WeightingStrategy",
    "RegularizationStrategy",
    "Optimizer",
    "StructuralModel",
]
