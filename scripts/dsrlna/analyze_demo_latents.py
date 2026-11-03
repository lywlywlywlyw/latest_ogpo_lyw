#!/usr/bin/env python3
"""Invert Square demos and export each DSRL-NA checkpoint's p(z'|s).

Run from anywhere with the project's Python, e.g.:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --max-samples 256
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --max-samples 0

Defaults select the run in dsrlna-online-20261102_201645.log. Re-running
includes newly saved params_*.pkl; --steps selects exact saved step numbers.
No training, simulator, wandb, or GPU is needed (CPU is the default).

Each stage NPZ contains z_demo, reconstruction diagnostics, base_loc/base_scale,
noise_mode, z_prime_samples [N,K,D], and log_prob_z_demo [N]. The exact law is
z' = action_magnitude * tanh(N(base_loc, diag(base_scale**2))). Its moments
are NOT base_loc/base_scale. noise_mode matches the agent's dist.mode().

The decoder uses dx/dt = v(s,x,t), forward Euler at t=i/flow_steps,
no intermediate clipping, and final [-1,1] clipping (DSRLNAAgent).
We integrate the SAME velocity field backwards with explicit Euler:
  x = demo_action
  for i = flow_steps, ..., 1:
      x = x - v(s,x,i/flow_steps) / flow_steps
  z_demo = x
The state s stays fixed. No clipping, optimization, or random initialization is
applied during inversion. Reverse Euler approximates the inverse continuous
flow; it is NOT an exact inverse of the finite-step forward Euler map. Final
clipping is non-injective: at a boundary we start at the observed action itself
and cannot recover its unknown pre-clipping value. We report boundary_action
and forward round-trip errors without using them to optimize z_demo.
Reference for reverse-time ODE integration (descending time_grid):
https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py

Only full within-episode chunks are used; terminal tails do not determine a
full decoder input. Out-of-actor-support latents have log probability -inf;
nonfinite inversions have NaN log probability. A large round-trip error is a
numerical diagnostic, not a reason to change the reverse trajectory.
Supports Square low_dim DSRL-NA with noise_chunk_length=-1 and encoder=None.

"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path, help='Directory containing flags.json and params_STEP.pkl')
    p.add_argument('--log', type=Path, default=ROOT / 'run_logs/dsrlna-online-20261102_201645.log')
    p.add_argument('--dataset', type=Path, default=Path('/root/datasets/square/mh/low_dim_v15.hdf5'))
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/dsrlna_demo_latents_reverse_euler')
    p.add_argument('--steps', type=int, nargs='+', help='Exact checkpoint filename steps (default: all currently saved)')
    p.add_argument('--max-samples', type=int, default=1024, help='Seeded subset; 0 means all full chunks')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-draws', type=int, default=32)
    p.add_argument('--tolerance', type=float, default=1e-4, help='Forward round-trip diagnostic threshold; does not affect inversion')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu')
    args = p.parse_args()
    if min(args.batch_size, args.num_draws) < 1 or args.max_samples < 0 or args.tolerance <= 0:
        p.error('batch-size, num-draws, tolerance must be positive; max-samples >= 0')
    return args


def discover_run(args):
    if args.run_dir:
        return args.run_dir.resolve()
    log = args.log.read_text()
    match = re.search(r'Online checkpoint directory \(when enabled\): (.+)', log)
    if not match:
        raise ValueError('Cannot identify run timestamp from log; pass --run-dir')
    stamp = Path(match[1].strip()).name
    candidates = list((ROOT / 'exp').glob(f'**/*{stamp}/flags.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one flags.json for {stamp}, found {candidates}; pass --run-dir')
    return candidates[0].parent


def load_demos(path, horizon, max_samples, seed):
    # Matches envs.robomimic_utils.get_dataset: numeric demo ordering,
    # low_dim key concatenation and float32 actions/observations, no normalization.
    import h5py
    import numpy as np
    # Same Square key order as TASK_CONFIG, without importing simulator packages.
    keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")

    observations, actions, demo_ids, timesteps = [], [], [], []
    with h5py.File(path, 'r') as f:
        for name in sorted(f['data'], key=lambda x: int(x[5:])):
            ep = f['data'][name]
            a = np.asarray(ep['actions'], dtype=np.float32)
            s = np.concatenate([np.asarray(ep[f'obs/{k}'], dtype=np.float32)
                                for k in keys], axis=-1)
            # Split also at any internal done: never invert masked/padded coordinates.
            done = np.asarray(ep['dones']).reshape(-1).astype(bool)
            starts = np.arange(max(0, len(a) - horizon + 1))
            starts = starts[np.array([not done[t:t + horizon - 1].any() for t in starts], dtype=bool)]
            observations.append(s[starts])
            actions.append(a[starts[:, None] + np.arange(horizon)].reshape(len(starts), horizon * a.shape[-1]))
            demo_ids.append(np.full(len(starts), int(name[5:]), dtype=np.int32))
            timesteps.append(starts)
    arrays = dict(observations=np.concatenate(observations), action_chunks=np.concatenate(actions),
                  demo_id=np.concatenate(demo_ids), timestep=np.concatenate(timesteps))
    total = len(arrays['observations'])
    if not total:
        raise ValueError('Dataset contains no full action chunks')
    idx = np.arange(total)
    if max_samples and max_samples < total:
        idx = np.sort(np.random.default_rng(seed).choice(total, max_samples, replace=False))
    return {k: v[idx] for k, v in arrays.items()}, total


def invert_flow_actions_ode(actor_fn, observations, actions, flow_steps, is_encoded=False):
    """Reverse-time explicit Euler for the velocity field used by the decoder.

    Forward: x[i+1] = x[i] + dt*v(s,x[i],i/K), i=0,...,K-1.
    Reverse: y[i-1] = y[i] - dt*v(s,y[i],i/K), i=K,...,1; y[K]=a.
    These approximate opposite directions of the same continuous ODE.
    Explicit Euler is not time-reversible at finite dt, so generally y[0]
    does not exactly undo the finite-step forward function. Neither the final
    clip nor intermediate clips have a unique inverse. DSRLNA disables
    intermediate clipping; its observed final clipped action is used as y[K].
    """
    import jax.numpy as jnp

    if flow_steps < 1:
        raise ValueError('flow_steps must be positive')
    batch_size = observations.shape[0]
    dt = 1.0 / flow_steps
    x = actions
    for i in range(flow_steps, 0, -1):
        # Current time is 1 on the first reverse step, then 1-dt, ..., dt.
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels = actor_fn(observations, x, t, is_encoded=is_encoded)
        x = x - vels * dt
    return x


def make_functions(config, dimension, args):
    import jax
    import jax.numpy as jnp
    from ogpo.networks import Actor, ActorVectorField
    from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
    from ogpo.agents.modules.pg_helper import sample_flow_actions_ode

    embedding = None
    if config.get('time_embedding', 'scalar') == 'sinusoidal':
        embedding = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))
    elif config.get('time_embedding', 'scalar') != 'scalar':
        raise ValueError('Unsupported time embedding')
    flow = ActorVectorField(hidden_dims=tuple(config.get('bc_flow_hidden_dims') or config['actor_hidden_dims']),
                            action_dim=dimension, layer_norm=config['actor_layer_norm'], time_embedding=embedding)
    magnitude = config['action_magnitude']
    actor = Actor(hidden_dims=tuple(config['actor_hidden_dims']), action_dim=dimension,
                  layer_norm=config['actor_layer_norm'], tanh_squash=True,
                  state_dependent_std=True, low=-magnitude, high=magnitude)
    steps = config['flow_steps']

    def velocity(params, s, x, t):
        return flow.apply({'params': params}, s, x, t, is_encoded=True)

    def forward(params, s, z):
        return sample_flow_actions_ode(
            lambda obs, x, t, is_encoded: velocity(params, obs, x, t), s, None,
            steps, dimension, act_min=-1., act_max=1., clip_intermediate=False,
            noises=z, is_encoded=True)

    @jax.jit
    def invert(params, states, actions):
        z = invert_flow_actions_ode(
            actor_fn=lambda obs, x, t, is_encoded: velocity(params, obs, x, t),
            observations=states, actions=actions, flow_steps=steps, is_encoded=True,
        )
        rebuilt = forward(params, states, z)
        errors = jnp.max(jnp.abs(rebuilt - actions), axis=-1)
        return z, rebuilt, errors

    @jax.jit
    def distribution(params, states, z, key):
        dist = actor.apply({'params': params}, states)
        samples = dist.sample(seed=key, sample_shape=(args.num_draws,)).swapaxes(0, 1)
        inside = jnp.all(jnp.abs(z) < magnitude, axis=-1)
        # Avoid atanh NaNs outside support; report the actual density as zero.
        safe_z = jnp.clip(z, -magnitude * (1 - 1e-6), magnitude * (1 - 1e-6))
        lp = jnp.where(inside, dist.log_prob(jnp.where(inside[:, None], z, safe_z)), -jnp.inf)
        return dict(base_loc=dist.distribution.mean(), base_scale=dist.distribution.stddev(),
                    noise_mode=dist.mode(), z_prime_samples=samples, log_prob_z_demo=lp,
                    actor_support_contains_z=inside)

    return invert, distribution


def main():
    args = parse_args()
    os.environ['JAX_PLATFORMS'] = args.platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    import numpy as np

    run = discover_run(args)
    flags = json.loads((run / 'flags.json').read_text())
    config = flags['agent']
    if (config['agent_name'] != 'dsrlna' or flags['env']['name'] != 'square-mh-low_dim'
            or config['encoder'] is not None or config.get('obs_two_tier', False)
            or config['noise_chunk_length'] != -1):
        raise ValueError('Requires Square low_dim DSRL-NA with encoder=None and noise_chunk_length=-1')
    paths = sorted(run.glob('params_*.pkl'), key=lambda p: int(p.stem.split('_')[-1]))
    if args.steps:
        selected = set(args.steps)
        paths = [p for p in paths if int(p.stem.split('_')[-1]) in selected]
        missing = selected - {int(p.stem.split('_')[-1]) for p in paths}
        if missing:
            raise FileNotFoundError(f'Checkpoint steps not yet saved: {sorted(missing)}')
    if not paths:
        raise FileNotFoundError(f'No params_*.pkl saved yet in {run}; rerun after the next save')
    horizon = config['horizon_length'] if config['action_chunking'] else 1
    data, available = load_demos(args.dataset, horizon, args.max_samples, args.seed)
    if not np.isfinite(data['observations']).all() or not np.isfinite(data['action_chunks']).all():
        raise ValueError('Nonfinite observations/actions in dataset')
    if np.any(np.abs(data['action_chunks']) > 1):
        raise ValueError('Demo actions outside decoder final clipping range [-1,1]')
    n, dimension = data['action_chunks'].shape
    invert, distribution = make_functions(config, dimension, args)
    out = args.output / run.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'demos.npz', **data)
    metadata = dict(run_dir=str(run), dataset=str(args.dataset.resolve()), available_full_chunks=available,
                    num_samples=n, horizon=horizon, seed=args.seed, num_draws=args.num_draws,
                    tolerance=args.tolerance, inverse_method='reverse_time_explicit_euler',
                    reverse_times=[i / config['flow_steps'] for i in range(config['flow_steps'], 0, -1)],
                    distribution='z_prime = action_magnitude * tanh(N(base_loc, diag(base_scale**2)))',
                    action_magnitude=config['action_magnitude'], flow_steps=config['flow_steps'],
                    boundary_semantics='Start at observed clipped action; unknown raw endpoint cannot be recovered', stages=[], skipped_checkpoints=[])
    cache = {}
    print(f'Run: {run}\nSamples: {n}/{available}; latent dimension: {dimension}; checkpoints: {len(paths)}', flush=True)
    for path in paths:
        # Full training checkpoints can contain large critics/optimizer states;
        # keep only inference weights and release the payload immediately.
        try:
            with path.open('rb') as f:
                payload = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as exc:
            if args.steps:
                raise
            print(f'Skipping incomplete checkpoint {path}: {exc}; rerun after saving finishes', flush=True)
            metadata['skipped_checkpoints'].append(str(path))
            continue
        state = payload['agent']
        flow_params = state['bc_flow_network']['params']['modules_actor_bc_flow']
        actor_params = state['actor_network']['params']['modules_actor']
        del state, payload
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(flow_params):
            digest.update(np.asarray(leaf).tobytes())
        fingerprint = digest.hexdigest()
        if fingerprint not in cache:
            batches = [invert(flow_params, data['observations'][i:i + args.batch_size],
                              data['action_chunks'][i:i + args.batch_size]) for i in range(0, n, args.batch_size)]
            z, reconstructed, error = [np.concatenate([np.asarray(b[j]) for b in batches]) for j in range(3)]
            cache = {fingerprint: dict(z_demo=z,
                                      reconstructed_actions=reconstructed, reconstruction_max_abs=error,
                                      inverse_finite=np.isfinite(z).all(-1),
                                      roundtrip_within_tolerance=np.isfinite(error) & (error <= args.tolerance),
                                      boundary_action=np.any(np.abs(data['action_chunks']) >= 1., axis=-1))}
        result = dict(cache[fingerprint])
        draws = []
        for i in range(0, n, args.batch_size):
            # Same random numbers at each stage for comparable Monte Carlo draws.
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), i)
            draws.append(jax.device_get(distribution(actor_params, data['observations'][i:i + args.batch_size],
                                                   result['z_demo'][i:i + args.batch_size], key)))
        result.update({k: np.concatenate([b[k] for b in draws]) for k in draws[0]})
        result['log_prob_z_demo'] = np.where(result['inverse_finite'], result['log_prob_z_demo'], np.nan)
        step = int(path.stem.split('_')[-1])
        filename = f'stage_{step}.npz'
        np.savez_compressed(out / filename, **result)
        summary = dict(step=step, checkpoint=str(path), output=filename, flow_sha256=fingerprint,
                       finite_fraction=float(result['inverse_finite'].mean()),
                       roundtrip_within_tolerance_fraction=float(result['roundtrip_within_tolerance'].mean()),
                       boundary_fraction=float(result['boundary_action'].mean()),
                       actor_support_fraction=float(result['actor_support_contains_z'].mean()),
                       reconstruction_max_abs=float(np.max(result['reconstruction_max_abs'])))
        metadata['stages'].append(summary)
        (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Step {step}: roundtrip_within_tolerance={summary["roundtrip_within_tolerance_fraction"]:.1%}, max error={summary["reconstruction_max_abs"]:.3g} -> {filename}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if not metadata['stages']:
        raise RuntimeError('No readable checkpoints; rerun after saving finishes')
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
