"""Reference baseline RMSNorm kernel in JAX for TPU execution."""
import jax
import jax.numpy as jnp

@jax.jit
def run_kernel(X: jnp.ndarray, W: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Standard un-fused RMSNorm implementation."""
    variance = jnp.mean(jnp.square(X), axis=-1, keepdims=True)
    rsqrt = jax.lax.rsqrt(variance + eps)
    return X * rsqrt * W
