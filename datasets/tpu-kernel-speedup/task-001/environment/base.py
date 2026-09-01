"""Reference baseline attention kernel in JAX for TPU execution."""
import jax
import jax.numpy as jnp

@jax.jit
def run_kernel(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, V)
    return output
