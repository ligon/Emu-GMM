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

# ruff: noqa: E402
# Imports below the float64 config update intentionally come after a
# non-import statement.
from importlib.metadata import PackageNotFoundError, version

# Enable float64 in JAX *before* any module-under-this-package import
# touches jax.numpy. JAX defaults to float32 (a deep-learning convention);
# for a numerical-statistics framework where Cholesky pivots, condition
# numbers, and gradient norms cross many orders of magnitude, float32
# precision is the wrong baseline. Notable symptoms with float32:
#   - optimistix LM cannot certify convergence at rtol=1e-8 because the
#     float32 noise floor on a O(0.1)-magnitude whitened residual is
#     around 2e-8;
#   - theta_hat itself drifts at the third significant digit relative to
#     the float64 solution.
# Users who explicitly want float32 can override after import with:
#     jax.config.update("jax_enable_x64", False)
import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Public re-exports.
from emu_gmm.covariance import (
    AnalyticalCovariance,
    ClusteredCovariance,
    IIDCovariance,
    SyntheticCovariance,
)
from emu_gmm.diagnostics import (
    build_diagnostics,
    build_optimizer_health,
    compute_cond_info,
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
    Emu_GMM_DimensionError,
    EstimationResult,
    Measure,
    Optimizer,
    OptimizerInfo,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import CUE, ContinuouslyUpdated, Fixed, Identity

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
    "CUE",
    # Regularization
    "DiagonalTikhonov",
    # Optimisers
    "optimistix_lm",
    "scipy_lm",
    # Result / diagnostics types
    "EstimationResult",
    "Diagnostics",
    "OptimizerInfo",
    # Errors
    "Emu_GMM_DimensionError",
    # Diagnostics builders
    "build_diagnostics",
    "build_optimizer_health",
    "compute_cond_info",
    # Protocols (for type-checking user code)
    "Measure",
    "CovarianceStrategy",
    "WeightingStrategy",
    "RegularizationStrategy",
    "Optimizer",
    "StructuralModel",
]
