"""Tests for emu_gmm.runtime: CPU host-config helpers (issue #115)."""

from __future__ import annotations

import os
import warnings

import emu_gmm
import pytest
from emu_gmm import runtime

_ENV_KEYS = (
    "JAX_NUM_CPU_DEVICES",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    runtime.SILENCE_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _isolate_env_and_latch():
    """Snapshot/restore the env vars and the warn-once latch around each test.

    ``configure`` mutates ``os.environ`` directly (not via monkeypatch), and
    ``maybe_warn_cpu_oversubscription`` flips a module global, so both are
    restored here to keep tests order-independent.
    """
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    saved_warned = runtime._warned
    yield
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    runtime._warned = saved_warned


class TestRecommendedEnv:
    def test_default_dict(self):
        assert emu_gmm.recommended_env() == {
            "JAX_NUM_CPU_DEVICES": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }

    def test_host_devices(self):
        env = emu_gmm.recommended_env(host_devices=2)
        assert env["JAX_NUM_CPU_DEVICES"] == "2"
        assert env["XLA_FLAGS"] == "--xla_force_host_platform_device_count=2"

    def test_thread_caps_false_drops_blas_vars(self):
        env = emu_gmm.recommended_env(thread_caps=False)
        assert set(env) == {"JAX_NUM_CPU_DEVICES", "XLA_FLAGS"}

    def test_invalid_host_devices_raises(self):
        with pytest.raises(ValueError, match="host_devices must be >= 1"):
            emu_gmm.recommended_env(host_devices=0)

    def test_pure_no_env_mutation(self):
        before = dict(os.environ)
        emu_gmm.recommended_env()
        assert dict(os.environ) == before


class TestConfigure:
    def test_raises_when_backend_already_initialized(self):
        # The JAX backend is initialised in this test process (importing
        # emu_gmm initialises it), so configure() must refuse rather than
        # silently no-op.
        with pytest.raises(RuntimeError, match="recommended_env"):
            emu_gmm.configure()

    def test_force_applies_even_when_initialized(self, monkeypatch):
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        applied = emu_gmm.configure(force=True)
        assert applied["JAX_NUM_CPU_DEVICES"] == "1"
        assert os.environ["JAX_NUM_CPU_DEVICES"] == "1"
        assert "--xla_force_host_platform_device_count=1" in os.environ["XLA_FLAGS"]
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_merges_existing_xla_flags(self, monkeypatch):
        monkeypatch.setenv(
            "XLA_FLAGS", "--foo=bar --xla_force_host_platform_device_count=8"
        )
        emu_gmm.configure(host_devices=1, force=True)
        flags = os.environ["XLA_FLAGS"]
        assert "--foo=bar" in flags  # unrelated flag preserved
        assert "--xla_force_host_platform_device_count=1" in flags
        assert "device_count=8" not in flags  # stale device cap replaced

    def test_apply_path_when_not_initialized(self, monkeypatch):
        # Simulate a fresh interpreter (backend not yet initialised): the
        # device cap is applied, no raise.
        monkeypatch.setattr(runtime, "_backend_initialized", lambda: False)
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        env = emu_gmm.configure()
        assert env["JAX_NUM_CPU_DEVICES"] == "1"
        assert os.environ["JAX_NUM_CPU_DEVICES"] == "1"

    def test_thread_caps_false(self, monkeypatch):
        monkeypatch.setattr(runtime, "_backend_initialized", lambda: False)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)
        emu_gmm.configure(thread_caps=False)
        assert "OMP_NUM_THREADS" not in os.environ


def _force_at_risk(monkeypatch, cores=64):
    monkeypatch.setattr(runtime, "_affinity_core_count", lambda: cores)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("JAX_NUM_CPU_DEVICES", raising=False)
    monkeypatch.delenv(runtime.SILENCE_ENV_VAR, raising=False)
    runtime._warned = False


def _assert_no_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn()  # must not raise -> no warning emitted


class TestWarning:
    def test_warns_on_many_core_uncapped_host(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=64)
        with pytest.warns(UserWarning, match="no JAX device cap"):
            runtime.maybe_warn_cpu_oversubscription()

    def test_warns_at_most_once(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=64)
        with pytest.warns(UserWarning):
            runtime.maybe_warn_cpu_oversubscription()
        _assert_no_warning(runtime.maybe_warn_cpu_oversubscription)

    def test_silenced_by_env(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=64)
        monkeypatch.setenv(runtime.SILENCE_ENV_VAR, "1")
        _assert_no_warning(runtime.maybe_warn_cpu_oversubscription)

    def test_no_warn_when_device_cap_in_xla_flags(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=64)
        monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
        _assert_no_warning(runtime.maybe_warn_cpu_oversubscription)

    def test_no_warn_when_jax_num_cpu_devices_set(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=64)
        monkeypatch.setenv("JAX_NUM_CPU_DEVICES", "1")
        _assert_no_warning(runtime.maybe_warn_cpu_oversubscription)

    def test_no_warn_on_small_host(self, monkeypatch):
        _force_at_risk(monkeypatch, cores=4)
        _assert_no_warning(runtime.maybe_warn_cpu_oversubscription)


class TestPublicApi:
    def test_exports(self):
        assert emu_gmm.configure is runtime.configure
        assert emu_gmm.recommended_env is runtime.recommended_env
        assert "configure" in emu_gmm.__all__
        assert "recommended_env" in emu_gmm.__all__


class TestEstimatorWiring:
    """``estimate`` / ``build_estimator`` call the one-time warning hook."""

    def _euler_measure(self):
        import jax.numpy as jnp
        from emu_gmm.examples.euler import euler_data

        x = euler_data(seed=0, n=60)
        return emu_gmm.EmpiricalMeasure(
            x=x, mask=jnp.ones((60, 3)), weights=jnp.ones(60)
        )

    def test_estimate_invokes_warning_hook(self, monkeypatch):
        import emu_gmm.estimator as est_mod
        from emu_gmm.examples.euler import EulerParams, euler_residual

        calls: list[int] = []
        monkeypatch.setattr(
            est_mod, "maybe_warn_cpu_oversubscription", lambda: calls.append(1)
        )
        emu_gmm.estimate(
            model=euler_residual,
            measure=self._euler_measure(),
            covariance=emu_gmm.IIDCovariance(),
            theta_init=EulerParams(beta=0.95, gamma=1.5),
        )
        assert calls, "estimate() did not call the #115 warning hook"

    def test_build_estimator_invokes_warning_hook(self, monkeypatch):
        import emu_gmm.estimator as est_mod
        from emu_gmm.examples.euler import EulerParams, euler_residual

        calls: list[int] = []
        monkeypatch.setattr(
            est_mod, "maybe_warn_cpu_oversubscription", lambda: calls.append(1)
        )
        emu_gmm.build_estimator(
            model=euler_residual,
            measure=self._euler_measure(),
            covariance=emu_gmm.IIDCovariance(),
            theta_init=EulerParams(beta=0.95, gamma=1.5),
        )
        assert calls, "build_estimator() did not call the #115 warning hook"
