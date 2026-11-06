"""One inverse update rule for agents, offline demos, and diagnostics.

The query grid reverses the forward decoder's evaluation times. This is an
explicit subtraction scheme, not the exact inverse of finite forward Euler
steps or of the decoder's final clipping.
"""

FLOW_INVERSE_METHOD = 'euler_subtraction_t_k_minus_1_to_zero_v1'
FLOW_INVERSE_OUTPUT_DIR = 'inverse_t_k_minus_1_to_zero'


def inverse_time_grid(flow_steps):
    """Return (K-1)/K, ..., 0; e.g. 0.9, ..., 0 for K=10."""
    if flow_steps < 1:
        raise ValueError('flow_steps must be positive')
    return tuple(i / flow_steps for i in range(flow_steps - 1, -1, -1))


def invert_flow_actions_ode(actor_fn, observations, actions, flow_steps, is_encoded=False):
    """Keep observations fixed and apply x <- x - v(s,x,t)/K exactly K times.

    Starts from the observed action vector. No clipping, optimization, random
    initialization, tanh/atanh, or other latent transformation is applied.
    Inputs are batched; agent entry points handle unbatched inputs explicitly.
    """
    import jax.numpy as jnp

    times = inverse_time_grid(flow_steps)
    dt = 1.0 / flow_steps
    x = actions
    for value in times:
        t = jnp.full((observations.shape[0], 1), value)
        velocity = actor_fn(observations, x, t, is_encoded=is_encoded)
        x = x - velocity * dt
    return x
