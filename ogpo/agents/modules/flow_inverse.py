"""Inverse update rules for agents, offline demos, and diagnostics.

The default Euler query grid reverses the forward decoder's evaluation times. This is an
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


TOP_INVERSE_STEPS = 10
TOP_INVERSE_NFE = 2 * TOP_INVERSE_STEPS


def invert_flow_actions_top(actor_fn, observations, actions, is_encoded=False, *,
                            start_time=0.95):
    """TOPS Eq. (3), Table 7: reverse flow with 10 explicit-midpoint steps.

    Default to start_time=0.95 (dt=0.095); pass 1.0 for the paper interval.
    Integrate dx/dt = v(s, x, t) from start_time to 0 in 10 equal steps. Each
    step queries the current state and its predicted midpoint (20 NFE total).
    Observations stay fixed, and neither intermediate states nor latents are
    clipped. This is flow reversal, not TOPS's schedule-aware forward solver.
    """
    import jax.numpy as jnp

    if not 0.0 < start_time <= 1.0:
        raise ValueError('start_time must lie in (0, 1]')
    dt = start_time / TOP_INVERSE_STEPS
    x = actions
    for i in range(TOP_INVERSE_STEPS):
        t = jnp.full((observations.shape[0], 1), start_time - i * dt)
        t_mid = jnp.full((observations.shape[0], 1), start_time - (i + 0.5) * dt)
        velocity = actor_fn(observations, x, t, is_encoded=is_encoded)
        midpoint = x - 0.5 * dt * velocity
        mid_velocity = actor_fn(observations, midpoint, t_mid, is_encoded=is_encoded)
        x = x - dt * mid_velocity
    return x


def invert_flow_actions_ode_trial(actor_fn, observations, actions, flow_steps,
                                  is_encoded=False, *, first_time=1.0,
                                  return_history=False):
    """Diagnostic reverse Euler on 1, (K-1)/K, ..., 1/K (K evaluations).

    first_time=0.99 replaces ONLY the first query; all later times and the
    subtraction step 1/K remain unchanged, isolating the endpoint effect.
    With return_history=True also return states including the input actions,
    shaped (K+1, batch, action_dim). No clipping is applied.
    """
    import jax.numpy as jnp

    inverse_time_grid(flow_steps)  # Validate before constructing the trial grid.
    if not 0.0 < first_time <= 1.0:
        raise ValueError('first_time must lie in (0, 1]')
    if flow_steps > 1 and first_time <= (flow_steps - 1) / flow_steps:
        raise ValueError('first_time must exceed the second query time')
    times = (first_time,) + tuple(i / flow_steps for i in range(flow_steps - 1, 0, -1))
    x = actions
    history = [x] if return_history else None
    for value in times:
        t = jnp.full((observations.shape[0], 1), value)
        velocity = actor_fn(observations, x, t, is_encoded=is_encoded)
        x = x - velocity / flow_steps
        if return_history:
            history.append(x)
    if return_history:
        return x, jnp.stack(history)
    return x
