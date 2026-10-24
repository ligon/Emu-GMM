"""Sample-reduction contracts anchored on ledger Section 26."""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
from emu_gmm import IIDCovariance, ReducingModel, estimate
from emu_gmm._internal.models import map_model_parameters, shift_model
from emu_gmm.covariance import (
    ClusteredCovariance,
    DesignAwareCovariance,
    StratifiedCovariance,
)
from emu_gmm.inference import k_statistic, profiled_k_confidence_set
from emu_gmm.measures import AnalyticalMeasure, EmpiricalMeasure, SyntheticMeasure
from emu_gmm.optimizer import optimistix_lm


@jdc.pytree_dataclass
class _Theta:
    a: float


class _Groups:
    """Two nonlinear group reductions with coordinate-specific populations."""

    def __init__(self):
        self.calls = 0

    def aux(self, measure, theta):
        self.calls += 1
        mass = measure.mask * measure.weights[:, None]
        # Guard excluded inputs before exp: aux sees original, un-imputed data.
        y = jnp.where(jnp.any(mass > 0, axis=1), measure.x[:, 0], 1.0)
        values = jnp.stack([jnp.exp(theta.a * y), theta.a * y**2], axis=1)
        ids = measure.x[:, 1].astype(jnp.int32)
        denom = jax.ops.segment_sum(mass, ids, num_segments=4)
        total = jax.ops.segment_sum(mass * values, ids, num_segments=4)
        return total / jnp.where(denom > 0, denom, 1.0)

    def psi(self, row, theta, aux):
        values = jnp.array([jnp.exp(theta.a * row[0]), theta.a * row[0] ** 2])
        return (values - aux[row[1].astype(jnp.int32)]) * row[2:4]

    def __call__(self, row, theta):
        raise AssertionError("the complete aux/psi pair must take precedence")


def _measure():
    rng = np.random.default_rng(113)
    n = 24
    y = rng.uniform(0.3, 1.8, n)
    groups = np.arange(n) % 3  # fourth declared group deliberately empty
    z = rng.normal(size=(n, 2))
    mask = np.ones((n, 2))
    mask[::4, 0] = 0
    mask[1::5, 1] = 0
    mask[-1] = 0
    y[-1] = np.nan
    w = rng.uniform(0.2, 1.8, n)
    w[0] = 0  # observed support remains mask-defined
    return EmpiricalMeasure(
        jnp.asarray(np.column_stack([y, groups, z])), jnp.asarray(mask), jnp.asarray(w)
    )


def _oracle(mu, a):
    x, mask, w = map(np.asarray, (mu.x, mu.mask, mu.weights))
    mass = mask * w[:, None]
    y = np.where(np.any(mass > 0, axis=1), x[:, 0], 1.0)
    values = np.column_stack([np.exp(a * y), a * y**2])
    deriv = np.column_stack([y * np.exp(a * y), y**2])
    groups = x[:, 1].astype(int)
    residual = np.empty_like(values)
    derivative = np.empty_like(values)
    for group in range(4):
        use = groups == group
        den = mass[use].sum(axis=0)
        safe_den = np.where(den > 0, den, 1)
        mean = (mass[use] * values[use]).sum(axis=0) / safe_den
        dmean = (mass[use] * deriv[use]).sum(axis=0) / safe_den
        residual[use] = (values[use] - mean) * x[use, 2:4]
        derivative[use] = (deriv[use] - dmean) * x[use, 2:4]
    g = mass * np.where(mask > 0, residual, 0)
    D = mass * np.where(mask > 0, derivative, 0)
    return g, D[..., None], mass.sum(axis=0)


def test_shared_pass_prepares_once_with_original_measure():
    model = _Groups()
    mu = _measure()
    theta = _Theta(0.3)
    m, safe, mass, Nj = mu.expectation_and_contributions(model, theta)
    assert model.calls == 1
    expected, _, expected_Nj = _oracle(mu, theta.a)
    np.testing.assert_allclose(m, expected.sum(0) / expected_Nj, atol=1e-14)
    np.testing.assert_allclose(mass * safe, expected, atol=1e-14)
    np.testing.assert_allclose(Nj, expected_Nj)
    assert isinstance(model, ReducingModel)


def test_complete_global_derivative_matches_independent_oracle_and_ad():
    mu, model, theta = _measure(), _Groups(), _Theta(0.3)
    g, D, Nj = _oracle(mu, theta.a)
    np.testing.assert_allclose(mu.moment_contributions(model, theta), g, atol=1e-14)
    np.testing.assert_allclose(mu.jacobian_contributions(model, theta), D, atol=1e-13)
    expected = D.sum(0) / Nj[:, None]
    np.testing.assert_allclose(mu.jacobian(model, theta), expected, atol=1e-13)
    automatic = jax.jacfwd(lambda a: mu.expectation(model, _Theta(a)))(theta.a)
    np.testing.assert_allclose(automatic, expected[:, 0], atol=1e-13)
    # Freezing aux before differentiation loses a material global term.
    fixed_aux = model.aux(mu, theta)

    def frozen(row, th):
        return model.psi(row, th, fixed_aux)

    wrong = mu.jacobian_contributions(frozen, theta)
    assert np.max(np.abs(np.asarray(wrong) - D)) > 0.1
    reverse = jax.grad(lambda a: jnp.sum(mu.expectation(model, _Theta(a))))(theta.a)
    np.testing.assert_allclose(reverse, expected.sum(), atol=1e-13)


def test_reducing_primitives_are_jit_vmap_clean_and_use_fresh_data():
    mu, model = _measure(), _Groups()
    fn = jax.jit(
        lambda a, measure: (
            measure.expectation(model, _Theta(a)),
            measure.jacobian_contributions(model, _Theta(a)),
        )
    )
    for a in (0.2, 0.5):
        mean, D = fn(a, mu)
        g_ref, d_ref, Nj = _oracle(mu, a)
        np.testing.assert_allclose(mean, g_ref.sum(0) / Nj, atol=1e-13)
        np.testing.assert_allclose(D, d_ref, atol=1e-13)
    batched = jax.jit(jax.vmap(lambda a: mu.expectation(model, _Theta(a))))(
        jnp.array([0.2, 0.5])
    )
    np.testing.assert_allclose(batched[1], fn(0.5, mu)[0], atol=1e-13)
    fresh = jdc.replace(mu, x=mu.x.at[:, 2:4].multiply(2.0))
    np.testing.assert_allclose(fn(0.2, fresh)[0], 2 * fn(0.2, mu)[0], atol=1e-13)
    changed = jdc.replace(
        mu,
        x=mu.x.at[:, 0].multiply(1.3),
        weights=mu.weights * jnp.linspace(0.4, 1.6, 24),
    )
    changed_mean, changed_D = fn(0.2, changed)
    g_ref, D_ref, Nj = _oracle(changed, 0.2)
    np.testing.assert_allclose(changed_mean, g_ref.sum(0) / Nj, atol=1e-13)
    np.testing.assert_allclose(changed_D, D_ref, atol=1e-13)
    old_aux = model.aux(mu, _Theta(0.2))

    def stale(row, th):
        return model.psi(row, th, old_aux)

    assert (
        np.max(
            np.abs(np.asarray(changed.expectation(stale, _Theta(0.2))) - changed_mean)
        )
        > 0.01
    )


def _covariances(fixed=False):
    psu = jnp.repeat(jnp.arange(12, dtype=float), 2)
    design = StratifiedCovariance(psu, jnp.floor(psu / 2), jnp.floor(psu / 4), 12, 6, 3)
    cluster = ClusteredCovariance(psu, 12, dof_correction=False)
    return [
        IIDCovariance(),
        IIDCovariance(centered=True),
        cluster,
        design,
        DesignAwareCovariance.from_design_mask(
            design, cluster, jnp.array([1.0, 0.0]), fixed_strata=fixed
        ),
    ]


@pytest.mark.parametrize("fixed", [False, True])
def test_standalone_cached_covariance_parity_and_one_aux_per_mixed_pass(fixed):
    mu, model, theta = _measure(), _Groups(), _Theta(0.3)
    for cov in _covariances(fixed):
        cache = mu.expectation_and_contributions(model, theta)
        expected = cov.covariance(model, theta, mu, cached_intermediates=cache)
        model.calls = 0
        actual = cov.covariance(model, theta, mu)
        assert model.calls == 1
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        # The independent row-residual matrix is also a valid ordinary oracle.
        _, safe, _, _ = cache
        row_ids = jnp.arange(mu.x.shape[0], dtype=float)
        oracle_mu = jdc.replace(mu, x=row_ids[:, None])

        def oracle(row, th, _safe=safe):
            return _safe[row[0].astype(jnp.int32)]

        np.testing.assert_allclose(
            actual, cov.covariance(oracle, theta, oracle_mu), atol=1e-14
        )


@pytest.mark.parametrize("kind", ["synthetic", "analytical"])
@pytest.mark.parametrize(
    "method",
    ["expectation", "jacobian", "moment_contributions", "jacobian_contributions"],
)
def test_unsupported_reducing_measure_paths_fail_explicitly(kind, method):
    if kind == "synthetic":
        mu = SyntheticMeasure(
            jax.random.PRNGKey(0), 4, lambda key, th: jnp.ones((4, 4))
        )
    else:
        mu = AnalyticalMeasure(
            lambda model, th: jnp.ones(2), lambda model, th: jnp.ones((2, 1))
        )
        if not hasattr(mu, method):
            return  # analytical measures have no observed-row contribution API
    with pytest.raises(TypeError, match="supported only on EmpiricalMeasure"):
        getattr(mu, method)(_Groups(), _Theta(0.3))


def test_incomplete_contract_has_clear_error():
    class Incomplete:
        def aux(self, measure, theta):
            return 0

    with pytest.raises(TypeError, match="both callable aux and psi"):
        _measure().expectation(Incomplete(), _Theta(0.3))


class _Labelled(_Groups):
    def psi(self, row, theta, aux):
        return ha.NamedArray(super().psi(row, theta, aux), (ha.Axis("Moments", 2),))


def test_estimator_label_probe_prepares_reducing_model():
    # The row model deliberately cannot be called without its reduction.
    mu = _measure()
    result = estimate(
        _Labelled(),
        mu,
        covariance=IIDCovariance(),
        parameters=_Theta(0.2),
        moment_names=("left", "right"),
        optimizer=optimistix_lm(max_steps=30),
    )
    assert result.labels.moment_names == ("left", "right")


def test_parameter_and_shift_adapters_recompute_aux_in_each_measure():
    mu, model = _measure(), _Groups()
    mapped = map_model_parameters(model, lambda th: _Theta(th.a * 2))
    np.testing.assert_allclose(
        mu.expectation(mapped, _Theta(0.2)), mu.expectation(model, _Theta(0.4))
    )
    shifted = shift_model(_Labelled(), jnp.array([0.2, -0.1]))
    for measure in (
        mu,
        jdc.replace(mu, weights=mu.weights * jnp.linspace(0.4, 1.6, 24)),
    ):
        np.testing.assert_allclose(
            measure.expectation(shifted, _Theta(0.2)),
            measure.expectation(model, _Theta(0.2)) - jnp.array([0.2, -0.1]),
            atol=1e-14,
        )


@jdc.pytree_dataclass
class _IVTheta:
    a: float
    b: float


class _IVReducing:
    def aux(self, measure, theta):
        return theta.b * jnp.mean(measure.x[:, 1])

    def psi(self, row, theta, aux):
        return (row[0] - theta.a * row[1] - theta.b - aux) * row[2:5]


def test_public_k_and_profile_match_full_derivative_closure_oracle():
    rng = np.random.default_rng(1131)
    z = np.column_stack([np.ones(80), rng.normal(size=(80, 2))])
    x = 1 + z[:, 1] + rng.normal(size=80) * 0.2
    y = 0.8 * x + 1.1 * (1 + x.mean()) + rng.normal(size=80) * 0.3
    mu = EmpiricalMeasure.from_arrays(jnp.asarray(np.column_stack([y, x, z])), M=3)
    model = _IVReducing()

    def oracle(row, theta):
        return (row[0] - theta.a * row[1] - theta.b * (1 + jnp.mean(mu.x[:, 1]))) * row[
            2:5
        ]

    theta = _IVTheta(0.8, 1.1)
    expected = k_statistic(theta, mu, IIDCovariance(), oracle)
    actual = k_statistic(theta, mu, IIDCovariance(), model)
    np.testing.assert_allclose(actual.K, expected.K, rtol=1e-10, atol=1e-12)
    kwargs = dict(
        measure=mu,
        covariance=IIDCovariance(),
        theta_builder=lambda a: _IVTheta(a, 1.1),
        profile=("a",),
        grid=np.array([0.65, 0.8, 0.95]),
    )
    native = profiled_k_confidence_set(model=model, **kwargs)
    reference = profiled_k_confidence_set(model=oracle, **kwargs)
    np.testing.assert_allclose(native.p_grid, reference.p_grid, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(
        native.nonconverged_indices, reference.nonconverged_indices
    )


def test_existing_row_callable_with_psi_method_remains_supported():
    class Existing:
        def psi(self, row, theta):
            return jnp.array([row[0] - theta.a, row[0] ** 2 - theta.a])

        __call__ = psi

    mu = _measure()
    np.testing.assert_allclose(
        mu.expectation(Existing(), _Theta(0.2)),
        mu.expectation(Existing().psi, _Theta(0.2)),
    )


@pytest.mark.parametrize("recenter", [False, True])
def test_public_refit_bootstrap_reduces_each_resampled_measure(recenter):
    from emu_gmm.inference import cluster_bootstrap

    class Observed(_IVReducing):
        def __init__(self):
            self.observed = []

        def aux(self, measure, theta):
            if not isinstance(measure.x, jax.core.Tracer):
                self.observed.append(float(jnp.mean(measure.x[:, 1])))
            return super().aux(measure, theta)

    rng = np.random.default_rng(1132)
    z = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    x = 1 + z[:, 1] + rng.normal(size=40) * 0.2
    y = 0.8 * x + 1.1 * (1 + x.mean()) + rng.normal(size=40) * 0.3
    mu = EmpiricalMeasure.from_arrays(jnp.asarray(np.column_stack([y, x, z])), M=3)
    model = Observed()
    covariance = ClusteredCovariance(
        jnp.arange(40, dtype=float), 40, dof_correction=False
    )
    options = {}
    if recenter:
        options["null_result"] = estimate(
            model, mu, covariance=covariance, parameters=_IVTheta(0.8, 1.1)
        )
    result = cluster_bootstrap(
        model,
        _IVTheta(0.8, 1.1),
        mu,
        covariance,
        n_boot=2,
        key=jax.random.PRNGKey(113),
        **options,
    )
    assert len(set(np.round(model.observed, 10))) >= 2
    assert result.theta_boot.array.shape == (2, 2)
    valid = np.asarray(result.convergence)
    np.testing.assert_array_equal(
        np.all(np.isfinite(np.asarray(result.theta_boot.array)), axis=1), valid
    )
