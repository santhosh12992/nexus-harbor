"""Reference baseline SwiGLU kernel in JAX for TPU execution."""
import jax
import jax.numpy as jnp

@jax.jit
def run_kernel(X: jnp.ndarray, W_gate: jnp.ndarray, W_up: jnp.ndarray) -> jnp.ndarray:
    """Standard un-fused SwiGLU implementation."""
    gate = jnp.matmul(X, W_gate)
    up = jnp.matmul(X, W_up)
    silu_gate = jax.nn.silu(gate)
    return silu_gate * up
