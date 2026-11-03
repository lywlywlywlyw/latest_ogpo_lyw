#!/usr/bin/env python3
"""Invert Square demos and score latents in each checkpoint's raw Gaussian.

Run from anywhere with the project's Python, e.g.:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --max-samples 256
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --max-samples 0

Defaults select the run in dsrlna-online-20261102_201645.log. Re-running
includes newly saved params_*.pkl; --steps selects exact saved step numbers.
No training, simulator, wandb, or GPU is needed (CPU is the default).

Relationship 1 is computed automatically: fixed demo latents are scored under
all checkpoint policies and compared against other-demonstration latents at
the SAME state. --num-negatives controls draws per state; --plot-relationship
writes PNG/PDF plots. --relationship-only reuses saved demo latents and reloads checkpoint actors. For example:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --plot-relationship
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --relationship-only --plot-relationship
See analyze_demo_policy_relationship() for exact support/tie conventions.

All scores directly evaluate the unchanged z_demo in the encoder's raw Gaussian:
log N(z_demo; base_loc, diag(base_scale**2)). No tanh, atanh, scaling, or
Jacobian correction is applied. This is a raw-Gaussian analysis score; the
trained policy still applies tanh when generating decoder inputs.
Each stage NPZ contains z_demo, reconstruction diagnostics, base_loc/base_scale,
noise_mode (Gaussian mean), z_prime_samples [N,K,D] (raw Gaussian samples), and
log_prob_z_demo [N]. Finite latents have no +/-action_magnitude restriction.

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
full decoder input. Nonfinite inversions have NaN log probability. A large round-trip error is a
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
    p.add_argument('--num-negatives', type=int, default=32, help='Other-demonstration latents per state for relationship 1')
    p.add_argument('--reference-step', type=int, help='Saved stage providing fixed demo latents (default: first analyzed stage)')
    p.add_argument('--plot-relationship', action='store_true', help='Plot relationship 1 versus checkpoint step as PNG/PDF')
    p.add_argument('--relationship-only', action='store_true', help='Reuse saved demo latents and reload checkpoint actors to compute log probabilities')
    p.add_argument('--tolerance', type=float, default=1e-4, help='Forward round-trip diagnostic threshold; does not affect inversion')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu')
    args = p.parse_args()
    if min(args.batch_size, args.num_draws, args.num_negatives) < 1 or args.max_samples < 0 or args.tolerance <= 0:
        p.error('batch-size, num-draws, num-negatives, tolerance must be positive; max-samples >= 0')
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


def latent_log_prob(dist, latents):
    """Score unchanged decoder-input latents in the raw, unsquashed Gaussian."""
    import jax.numpy as jnp

    log_prob = dist.log_prob(latents)
    finite = jnp.all(jnp.isfinite(latents), axis=-1)
    # A nondegenerate Gaussian has support on all finite real vectors.
    return jnp.where(finite, log_prob, jnp.nan), finite


def make_noise_actor(config, dimension):
    """Load checkpoint Gaussian heads with tanh disabled for this analysis."""
    from ogpo.networks import Actor

    return Actor(hidden_dims=tuple(config['actor_hidden_dims']), action_dim=dimension,
                 layer_norm=config['actor_layer_norm'], tanh_squash=False,
                 state_dependent_std=True)


def make_actor_latent_scorer(config, dimension):
    """Score matched and other-demo latents directly in the actor's output distribution."""
    import jax
    import jax.numpy as jnp

    actor = make_noise_actor(config, dimension)

    @jax.jit
    def score(params, states, z_demo, z_other):
        dist = actor.apply({'params': params}, states)
        matched_lp, matched_support = latent_log_prob(dist, z_demo)
        # dist has batch shape [B]; leading K is a sample dimension.
        other_lp, other_support = latent_log_prob(
            dist, jnp.swapaxes(z_other, 0, 1))
        return matched_lp, other_lp.T, matched_support, other_support.T

    return score


def sample_cross_demo_indices(demo_ids, num_negatives=32, seed=1):
    """Uniformly sample latent rows from OTHER demonstrations, with replacement.

    Each state gets K independent draws from all rows with a different demo_id.
    Longer demos contribute more candidate rows. The same fixed indices are
    reused at every checkpoint. This excludes an entire trajectory, not just i.
    """
    import numpy as np

    demo_ids = np.asarray(demo_ids)
    if demo_ids.ndim != 1 or num_negatives < 1:
        raise ValueError('demo_ids must be a vector and num_negatives >= 1')
    groups, inverse, counts = np.unique(demo_ids, return_inverse=True, return_counts=True)
    if len(groups) < 2:
        raise ValueError('Matching analysis requires samples from at least two different demonstrations; increase --max-samples')
    order = np.argsort(demo_ids, kind='stable')
    starts = np.cumsum(counts) - counts
    block_start = starts[inverse, None]
    block_count = counts[inverse, None]
    ranks = np.random.default_rng(seed).integers(
        0, len(demo_ids) - block_count, size=(len(demo_ids), num_negatives))
    return order[ranks + np.where(ranks >= block_start, block_count, 0)]


def analyze_demo_policy_relationship(output_dir, *, num_negatives=32, seed=1,
                                     steps=None, reference_step=None, plot=False, batch_size=32):
    """Relationship 1: L_t(s_i) and cross-demo conditional matching probability.

    Reloads each checkpoint's Gaussian heads with tanh_squash=False and calls
    gaussian.log_prob(z_demo) directly, with no transformation of z_demo.
    Saved base_loc/base_scale are not used. This scores the raw Gaussian,
    rather than the trained policy's tanh-transformed decoder-input density.
    Saved stage_*.npz files supply only the fixed reference demo latents.
    z_d is FIXED at reference_step (default: first selected stage), even if the
    decoder changes later. All stages use the same states and negative indices.
    For each i, compare pi_t(z_di|s_i) against K latents from other demo IDs.

    strict_match_probability averages 1[L_match > L_negative] over pairs with
    defined log probabilities. -inf/-inf is a tie and contributes zero wins;
    NaNs are excluded and counted. Also report finite/support-conditioned win
    rate, ties, support coverage, and the comparison coverage. These are density
    comparisons, not action likelihoods or posterior probabilities.

    Writes one relationship_stage_STEP.npz per stage (including all L values),
    relationship_reference.npz, and demo_policy_relationship.json. plot=True
    also writes a PNG and PDF through plot_demo_policy_relationship().
    """
    import jax
    import numpy as np

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / 'manifest.json').read_text())
    stages = sorted(manifest['stages'], key=lambda stage: stage['step'])
    if steps is not None:
        selected = set(steps)
        stages = [stage for stage in stages if stage['step'] in selected]
        missing = selected - {stage['step'] for stage in stages}
        if missing:
            raise ValueError(f'Stages missing from manifest: {sorted(missing)}')
    if not stages:
        raise ValueError('No saved stages to analyze')
    reference_step = stages[0]['step'] if reference_step is None else reference_step
    known_stages = {stage['step']: stage for stage in manifest['stages']}
    if reference_step not in known_stages:
        raise ValueError(f'Reference step {reference_step} is absent from this manifest; analyze it together with the target steps')
    with np.load(output_dir / f'stage_{reference_step}.npz') as reference:
        z_demo = np.asarray(reference['z_demo']).copy()
    with np.load(output_dir / 'demos.npz') as demos:
        demo_ids = np.asarray(demos['demo_id']).copy()
        observations = np.asarray(demos['observations']).copy()
    if z_demo.ndim != 2 or len(z_demo) != len(demo_ids):
        raise ValueError('Reference latents and saved demo rows do not match')
    negatives = sample_cross_demo_indices(demo_ids, num_negatives, seed)
    np.savez_compressed(output_dir / 'relationship_reference.npz', z_demo=z_demo,
                        demo_id=demo_ids, negative_indices=negatives, reference_step=reference_step)
    run_dir = Path(manifest['run_dir'])
    config = json.loads((run_dir / 'flags.json').read_text())['agent']
    magnitude = config['action_magnitude']
    score_latents = make_actor_latent_scorer(config, z_demo.shape[-1])
    records = []

    def mean(values, mask=None):
        values = np.asarray(values)
        if mask is not None:
            values = values[mask]
        return float(np.mean(values)) if values.size else float('nan')

    def json_number(value):
        # Keep JSON standard-compliant and distinguish zero density from missing.
        if np.isnan(value):
            return None
        if np.isneginf(value):
            return '-inf'
        if np.isposinf(value):
            return '+inf'
        return float(value)

    for stage in stages:
        checkpoint = Path(stage.get('checkpoint', run_dir / f'params_{stage["step"]}.pkl'))
        with checkpoint.open('rb') as stream:
            payload = pickle.load(stream)
        actor_params = payload['agent']['actor_network']['params']['modules_actor']
        del payload
        matched_lp = np.empty(len(z_demo), dtype=np.float64)
        matched_support = np.empty(len(z_demo), dtype=bool)
        negative_lp = np.empty(negatives.shape, dtype=np.float64)
        negative_support = np.empty(negatives.shape, dtype=bool)
        for start in range(0, len(z_demo), batch_size):
            end = min(start + batch_size, len(z_demo))
            values = jax.device_get(score_latents(
                actor_params, observations[start:end], z_demo[start:end],
                z_demo[negatives[start:end]]))
            matched_lp[start:end], negative_lp[start:end], matched_support[start:end], negative_support[start:end] = values
        del actor_params
        defined = ~np.isnan(matched_lp[:, None]) & ~np.isnan(negative_lp)
        both_support = defined & matched_support[:, None] & negative_support
        finite_pairs = np.isfinite(matched_lp[:, None]) & np.isfinite(negative_lp)
        wins = defined & (matched_lp[:, None] > negative_lp)
        ties = defined & (matched_lp[:, None] == negative_lp)
        losses = defined & (matched_lp[:, None] < negative_lp)
        defined_count = defined.sum(axis=1)
        per_state = np.divide(wins.sum(axis=1), defined_count,
                              out=np.full(len(z_demo), np.nan), where=defined_count > 0)
        filename = f'relationship_stage_{stage["step"]}.npz'
        np.savez_compressed(output_dir / filename, matched_log_prob=matched_lp,
                            negative_log_prob=negative_lp, matched_in_support=matched_support,
                            negative_in_support=negative_support, comparison_defined=defined,
                            strict_match_per_state=per_state)
        numeric = dict(
            strict_match_probability=mean(wins, defined), tie_probability=mean(ties, defined),
            loss_probability=mean(losses, defined),
            tie_adjusted_match_probability=mean(wins.astype(float) + .5 * ties, defined),
            both_in_support_match_probability=mean(wins, both_support),
            finite_pair_match_probability=mean(wins, finite_pairs),
            defined_pair_fraction=mean(defined), both_in_support_pair_fraction=mean(both_support),
            matched_support_fraction=mean(matched_support), negative_support_fraction=mean(negative_support),
            matched_log_prob_mean=mean(matched_lp, ~np.isnan(matched_lp)),
            negative_log_prob_mean=mean(negative_lp, ~np.isnan(negative_lp)),
            matched_log_prob_finite_mean=mean(matched_lp, np.isfinite(matched_lp)),
            negative_log_prob_finite_mean=mean(negative_lp, np.isfinite(negative_lp)),
            matched_log_prob_finite_median=float(np.median(matched_lp[np.isfinite(matched_lp)]))
            if np.isfinite(matched_lp).any() else float('nan'))
        record = dict(step=stage['step'], output=filename, flow_sha256=stage.get('flow_sha256'), num_pairs=int(defined.size),
                      num_defined_pairs=int(defined.sum()), num_both_in_support_pairs=int(both_support.sum()),
                      **{key: json_number(value) for key, value in numeric.items()})
        records.append(record)
        print(f'Relationship step {stage["step"]}: P(match > other-demo)={numeric["strict_match_probability"]:.3f}, '
              f'ties={numeric["tie_probability"]:.3f}, matched support={numeric["matched_support_fraction"]:.1%}', flush=True)
    summary = dict(
        relationship='L_t(s_i)=log N(z_di; mu_t(s_i), diag(sigma_t(s_i)**2)); matched vs other-demo',
        log_prob_source='checkpoint raw Gaussian (tanh_squash=False): dist.log_prob(unmodified_z_demo)',
        distribution='raw_gaussian', support='All finite real latent vectors; no action_magnitude restriction',
        reference_step=int(reference_step), reference_flow_sha256=known_stages[reference_step].get('flow_sha256'),
        fixed_demo_latents=True,
        num_states=len(z_demo), num_negatives=num_negatives, seed=seed,
        negative_sampling='Uniform latent row from a different demo_id, with replacement; fixed across stages',
        action_magnitude=magnitude, strict_comparison='Ties, including -inf/-inf, are not wins; NaN pairs excluded',
        json_nonfinite='null means undefined; -inf string means zero density',
        latent_provenance='reverse-time Euler; decoder final clipping and discretization affect z_demo',
        stages=records)
    summary_path = output_dir / 'demo_policy_relationship.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if plot:
        plot_demo_policy_relationship(summary_path)
    return summary


def plot_demo_policy_relationship(summary_path, output_prefix=None):
    """Plot saved relationship metrics versus checkpoint step; no model loading."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    stages = summary['stages']
    x = [stage['step'] for stage in stages]

    def series(key):
        return np.array([float(stage[key]) if stage[key] is not None else np.nan for stage in stages])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    axes[0].plot(x, series('matched_log_prob_finite_mean'), 'o-', label='Matched demo latent')
    axes[0].plot(x, series('negative_log_prob_finite_mean'), 'o-', label='Other-demo latent at same state')
    axes[0].set_ylabel('Mean finite log density')
    axes[0].set_title('Raw Gaussian scores on unchanged demo latents (no tanh)'
                      if summary.get('distribution') == 'raw_gaussian'
                      else 'Finite densities only; zero-density coverage is shown below')
    axes[1].plot(x, series('strict_match_probability'), 'o-', label='P(matched > other-demo), defined pairs')
    axes[1].plot(x, series('both_in_support_match_probability'), 'o-', label='P(matched > other-demo | both in support)')
    axes[1].plot(x, series('tie_probability'), ':', label='Tie fraction (includes -inf / -inf)')
    axes[1].axhline(.5, color='gray', linestyle='--', label='0.5 reference; strict wins depend on ties')
    axes[1].set_ylabel('Matching probability')
    axes[2].plot(x, series('matched_support_fraction'), 'o-', label='Matched latent in support')
    axes[2].plot(x, series('negative_support_fraction'), 'o-', label='Other-demo latent in support')
    axes[2].plot(x, series('both_in_support_pair_fraction'), ':', label='Both latents in support')
    axes[2].plot(x, series('defined_pair_fraction'), '--', label='Defined comparison fraction')
    axes[2].set_ylabel('Fraction')
    axes[2].set_xlabel('Checkpoint step')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_ylim(-.03, 1.03)
    label = 'encoder raw Gaussian' if summary.get('distribution') == 'raw_gaussian' else 'conditional policy'
    fig.suptitle(f'Demo latent / {label} relationship (fixed z_d from step {summary["reference_step"]})')
    prefix = Path(output_prefix) if output_prefix is not None else summary_path.with_suffix('')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(str(prefix) + '.' + extension, dpi=200)
    plt.close(fig)
    return [Path(str(prefix) + '.' + extension) for extension in ('png', 'pdf')]


def make_functions(config, dimension, args):
    import jax
    import jax.numpy as jnp
    from ogpo.networks import ActorVectorField
    from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
    from ogpo.agents.modules.pg_helper import sample_flow_actions_ode

    embedding = None
    if config.get('time_embedding', 'scalar') == 'sinusoidal':
        embedding = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))
    elif config.get('time_embedding', 'scalar') != 'scalar':
        raise ValueError('Unsupported time embedding')
    flow = ActorVectorField(hidden_dims=tuple(config.get('bc_flow_hidden_dims') or config['actor_hidden_dims']),
                            action_dim=dimension, layer_norm=config['actor_layer_norm'], time_embedding=embedding)
    actor = make_noise_actor(config, dimension)
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
        lp, inside = latent_log_prob(dist, z)
        return dict(base_loc=dist.mean(), base_scale=dist.stddev(),
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
    out = args.output / run.name
    if args.relationship_only:
        analyze_demo_policy_relationship(out, num_negatives=args.num_negatives, seed=args.seed,
                                         steps=args.steps, reference_step=args.reference_step,
                                         plot=args.plot_relationship, batch_size=args.batch_size)
        print(f'Relationship results: {out}', flush=True)
        return
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
                    distribution='raw Gaussian: z_prime ~ N(base_loc, diag(base_scale**2)); no tanh',
                    log_prob_source='raw Gaussian log_prob(unmodified_z_demo)',
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
    analyze_demo_policy_relationship(out, num_negatives=args.num_negatives, seed=args.seed,
                                     reference_step=args.reference_step, plot=args.plot_relationship, batch_size=args.batch_size)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
