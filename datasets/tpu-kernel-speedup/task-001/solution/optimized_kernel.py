"""Optimized fused attention kernel in JAX for TPU execution."""
import jax
import jax.numpy as jnp

@jax.jit
def run_kernel(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.float32(1.0 / jnp.sqrt(Q.shape[-1]))
    scores = jnp.einsum('bhqd,bhkd->bhqk', Q * scale, K)
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', weights, V)
    return output
