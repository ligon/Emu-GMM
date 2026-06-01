"""emu-gmm: Measure-theoretic GMM. Estimation via E_mu.

The public API is re-exported here for convenience::

    from emu_gmm import (
        estimate,
        SyntheticMeasure, SyntheticCovariance,
        AnalyticalMeasure, AnalyticalCovariance,
        EmpiricalMeasure, IIDCovariance, ClusteredCovariance,
        Identity, Fixed, ContinuouslyUpdated, IteratedWeighting,
        DiagonalTikhonov,
        TikhonovPenalty,
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
    DesignAwareCovariance,
    IIDCovariance,
    StratifiedCovariance,
    SumCovariance,
    SyntheticCovariance,
)
from emu_gmm.diagnostics import (
    build_diagnostics,
    build_optimizer_health,
    compute_cond_info,
)
from emu_gmm.estimator import build_estimator, estimate
from emu_gmm.inference import (
    ClusterBootstrapResult,
    JTestResult,
    KStatisticResult,
    WildBootstrapResult,
    cluster_bootstrap,
    j_test,
    k_statistic,
    moment_wild_bootstrap,
)
from emu_gmm.manifolds import (
    Euclidean,
    ManifoldLeaf,
    ManifoldParam,
    Positive,
    Product,
    PSDFixedRank,
    riemannian_lm,
)
from emu_gmm.measures import (
    AnalyticalMeasure,
    EmpiricalMeasure,
    SyntheticMeasure,
)
from emu_gmm.numerics import ridge_inverse
from emu_gmm.optimizer import linear_solver, optimistix_lm, scipy_lm
from emu_gmm.parameter_space import ParameterSpace, on
from emu_gmm.penalty import PenaltyStrategy, TikhonovPenalty
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    Diagnostics,
    Emu_GMM_DimensionError,
    EstimationResult,
    LabelContext,
    Measure,
    Optimizer,
    OptimizerInfo,
    RegularizationStrategy,
    StructuralModel,
    WeightingStrategy,
)
from emu_gmm.weighting import (
    CUE,
    ContinuouslyUpdated,
    Fixed,
    Identity,
    IteratedWeighting,
)

try:
    __version__ = version("emu-gmm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Entry point
    "estimate",
    "build_estimator",
    # Inference
    "k_statistic",
    "KStatisticResult",
    # Measures
    "SyntheticMeasure",
    "AnalyticalMeasure",
    "EmpiricalMeasure",
    # Covariance strategies
    "SyntheticCovariance",
    "AnalyticalCovariance",
    "IIDCovariance",
    "ClusteredCovariance",
    "StratifiedCovariance",
    "DesignAwareCovariance",
    "SumCovariance",
    # Weighting strategies
    "Identity",
    "Fixed",
    "ContinuouslyUpdated",
    "CUE",
    "IteratedWeighting",
    # Regularization
    "DiagonalTikhonov",
    # Numerics helpers
    "ridge_inverse",
    # Penalty (in-objective parameter penalty)
    "TikhonovPenalty",
    # Optimisers
    "optimistix_lm",
    "scipy_lm",
    "linear_solver",
    "riemannian_lm",
    # Inference helpers
    "cluster_bootstrap",
    "ClusterBootstrapResult",
    "j_test",
    "JTestResult",
    # Result / diagnostics types
    "EstimationResult",
    "Diagnostics",
    "OptimizerInfo",
    "LabelContext",
    # Errors
    "Emu_GMM_DimensionError",
    # Diagnostics builders
    "build_diagnostics",
    "build_optimizer_health",
    "compute_cond_info",
    # Inference
    "moment_wild_bootstrap",
    "WildBootstrapResult",
    # Manifolds (parameter geometry — a first-class problem-tuple menu, peer
    # to Measures / Covariance strategies; wrap a non-scalar parameter leaf in
    # ManifoldLeaf(array, manifold) and estimate via the ordinary entry point.
    # ManifoldSpec / LeafSpec stay submodule-internal — they are machinery,
    # not part of the user-composed tuple.)
    "Euclidean",
    "PSDFixedRank",
    "Product",
    "Positive",
    "ManifoldLeaf",
    # Parameter-space declaration layer (#107): declare field -> manifold
    # geometry once in a class, then ParameterSpace.point([seed]) lowers to a
    # ManifoldLeaf pytree (a valid theta_init) consumed by estimate(parameters=).
    "ParameterSpace",
    "on",
    # Protocols (for type-checking user code)
    "Measure",
    "CovarianceStrategy",
    "WeightingStrategy",
    "RegularizationStrategy",
    "PenaltyStrategy",
    "Optimizer",
    "StructuralModel",
    "ManifoldParam",
]
