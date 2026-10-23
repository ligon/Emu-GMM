"""Compile-time memory probe for the bounded Jacobian option (#232).

Run with CPU affinity and a low nice priority, for example::

    nice -n 19 taskset -c 40 python scripts/validation/jacobian_memory_232.py

The table reports XLA's temporary-buffer estimate, not observed peak RSS.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from emu_gmm._internal.jacobian import columnwise_jacobian
from emu_gmm.optimizer import optimistix_lm


def _compile_sizes(k: int, n: int) -> list[tuple[str, int, int]]:
    grid = jnp.linspace(0.01, 1.0, n)
    start = jnp.linspace(0.2, 0.7, k)

    def residual(theta):
        values = jnp.sin(grid[:, None] * theta[None, :])
        return jnp.concatenate(
            [jnp.mean(values, axis=0) - 0.1, jnp.mean(values**2, axis=0) - 0.05]
        )

    cases = {
        "postfit-jacfwd": jax.jit(jax.jacfwd(residual)),
        "postfit-columnwise": jax.jit(lambda x: columnwise_jacobian(residual, x)),
        "lm-qr": jax.jit(lambda x: optimistix_lm(max_steps=3)(residual, x)[0]),
        "lm-lsmr": jax.jit(
            lambda x: optimistix_lm(max_steps=3, jacobian_memory="bounded")(
                residual, x
            )[0]
        ),
    }
    rows = []
    for name, fn in cases.items():
        compiled = fn.lower(start).compile()
        analysis = compiled.memory_analysis()
        rows.append(
            (
                name,
                int(analysis.temp_size_in_bytes),
                int(analysis.output_size_in_bytes),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--k", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    print(f"backend={jax.default_backend()} x64={jax.config.jax_enable_x64} n={args.n}")
    print("k method temp_bytes output_bytes")
    for k in args.k:
        for name, temp, output in _compile_sizes(k, args.n):
            print(k, name, temp, output)


if __name__ == "__main__":
    main()
