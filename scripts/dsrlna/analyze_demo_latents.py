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

Per-trajectory clustering (latest checkpoint by default):
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --cluster-trajectories
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --cluster-trajectories --demo-ids 10 42 --steps 2000000
This mode ignores --max-samples and inverts ALL valid full chunks of each
selected trajectory. KMeans operates in the original latent space, trying
K=2..--cluster-max-k; PCA only supplies a 2D display plane.

Joint clustering of ALL Square demos:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --cluster-all-trajectories --steps 2000000 --batch-size 256
Fits K=2..20 on all raw z, and separately reports an explicitly labeled
sensitivity analysis excluding extreme norms (>10 times the median).

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
    p.add_argument('--platform', choices=['cpu', 'gpu', 'cuda'], default='cpu')
    p.add_argument('--cluster-trajectories', action='store_true', help='Invert complete trajectories and cluster each independently; default latest checkpoint')
    p.add_argument('--num-trajectories', type=int, default=8)
    p.add_argument('--demo-ids', type=int, nargs='+', help='Explicit trajectory IDs instead of seeded random selection')
    p.add_argument('--cluster-max-k', type=int, default=10)
    p.add_argument('--temporal-clusters-from', type=Path, help='Analyze contiguous latent clouds from an existing trajectory_clusters directory; no reinversion')
    p.add_argument('--temporal-permutations', type=int, default=499)
    p.add_argument('--cluster-all-trajectories', action='store_true', help='Invert all demos and cluster all raw z jointly; latest checkpoint by default')
    p.add_argument('--global-max-k', type=int, default=20)
    p.add_argument('--global-eval-samples', type=int, default=3000)
    args = p.parse_args()
    if args.global_max_k < 2 or args.global_eval_samples < 100:
        p.error('global-max-k >= 2 and global-eval-samples >= 100 required')
    if args.temporal_permutations < 19:
        p.error('temporal-permutations must be >= 19')
    if sum((args.relationship_only, args.cluster_trajectories, args.cluster_all_trajectories, bool(args.temporal_clusters_from))) > 1:
        p.error('relationship-only, cluster-trajectories, cluster-all-trajectories and temporal-clusters-from are mutually exclusive')
    if args.num_trajectories < 1 or args.cluster_max_k < 2:
        p.error('num-trajectories >= 1 and cluster-max-k >= 2 required')
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


def load_demos(path, horizon, max_samples, seed, selected_demo_ids=None):
    # Matches envs.robomimic_utils.get_dataset: numeric demo ordering,
    # low_dim key concatenation and float32 actions/observations, no normalization.
    import h5py
    import numpy as np
    # Same Square key order as TASK_CONFIG, without importing simulator packages.
    keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")

    observations, actions, demo_ids, timesteps = [], [], [], []
    with h5py.File(path, 'r') as f:
        for name in sorted(f['data'], key=lambda x: int(x[5:])):
            if selected_demo_ids is not None and int(name[5:]) not in selected_demo_ids:
                continue
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


def analyze_trajectory_latent_clusters(args, run, config, paths):
    """Invert every valid full chunk in selected demos, then cluster in raw z space.

    No transition subsampling, standardization, or PCA before KMeans. K=1 is
    retained as the inertia baseline; silhouette can only rank K>=2 and cannot
    establish that discrete clusters exist. PCA is fitted separately per demo
    for visualization only. Overlapping chunks are temporally correlated.
    """
    import csv
    import h5py
    import jax
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score, adjusted_rand_score)
    from threadpoolctl import threadpool_limits

    horizon = config['horizon_length'] if config['action_chunking'] else 1
    with h5py.File(args.dataset, 'r') as f:
        lengths = {int(name[5:]): len(f['data'][name]['actions']) for name in f['data']}
    eligible = sorted(i for i, length in lengths.items() if length >= horizon + 2)
    ids = sorted(set(args.demo_ids)) if args.demo_ids else sorted(
        np.random.default_rng(args.seed).choice(eligible, min(args.num_trajectories, len(eligible)), replace=False).tolist())
    if not ids or set(ids) - set(eligible):
        raise ValueError('Requested demos missing or too short for clustering')
    data, _ = load_demos(args.dataset, horizon, 0, args.seed, selected_demo_ids=set(ids))
    if not all(np.isfinite(data[k]).all() for k in ('observations', 'action_chunks')):
        raise ValueError('Nonfinite observations/actions')
    if np.any(np.abs(data['action_chunks']) > 1):
        raise ValueError('Actions outside [-1,1]')
    invert, _ = make_functions(config, data['action_chunks'].shape[1], args)
    selected_paths = paths if args.steps else paths[-1:]
    for path in selected_paths:
        step = int(path.stem.split('_')[-1])
        out = args.output / run.name / f'trajectory_clusters_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as f:
            payload = pickle.load(f)
        params = payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow']
        del payload
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(params):
            digest.update(np.asarray(leaf).tobytes())
        summary = dict(checkpoint=str(path), flow_sha256=digest.hexdigest(), dataset=str(args.dataset),
                       seed=args.seed, selected_demo_ids=ids, horizon=horizon,
                       latent_dimension=data['action_chunks'].shape[1], flow_steps=config['flow_steps'],
                       inverse_method='reverse_time_explicit_euler; same invert as existing analysis',
                       clustering='KMeans in raw latent space, n_init=20, no scaling; 3 independent seeds per K',
                       selection='maximum full-data silhouette over K=2..max_k; not evidence against K=1',
                       projection='per-trajectory PCA; visualization only', trajectories=[])
        overview, overview_axes = plt.subplots(len(ids), 2, figsize=(12, 3.6 * len(ids)), squeeze=False)
        for row, demo_id in enumerate(ids):
            mask = data['demo_id'] == demo_id
            states, actions, times = (data[k][mask] for k in ('observations', 'action_chunks', 'timestep'))
            n = len(times)
            batches = []
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                # Pad final batch to avoid a separate JAX compilation per demo.
                pad = args.batch_size - (end - start)
                s_batch = np.pad(states[start:end], ((0, pad), (0, 0)), mode='edge')
                a_batch = np.pad(actions[start:end], ((0, pad), (0, 0)), mode='edge')
                batches.append(tuple(np.asarray(x)[:end-start] for x in invert(params, s_batch, a_batch)))
            z, rebuilt, errors = [np.concatenate([b[j] for b in batches]) for j in range(3)]
            if not np.isfinite(z).all():
                raise ValueError(f'demo_{demo_id}: nonfinite inversion; cannot cluster every transition')
            metrics, labels_all = [], {}
            # Limit BLAS threads: tiny per-demo matrices otherwise oversubscribe CPUs.
            with threadpool_limits(limits=1):
                baseline = KMeans(n_clusters=1, n_init=1, random_state=args.seed).fit(z).inertia_
                for k in range(2, min(args.cluster_max_k, n - 1) + 1):
                    fits = [KMeans(n_clusters=k, n_init=20, random_state=args.seed + repeat).fit(z)
                            for repeat in range(3)]
                    best = min(fits, key=lambda fit: fit.inertia_)
                    labels = best.labels_
                    if len(np.unique(labels)) != k:
                        continue
                    labels_all[f'labels_k{k}'] = labels
                    stability = [adjusted_rand_score(fits[a].labels_, fits[b].labels_)
                                 for a, b in ((0, 1), (0, 2), (1, 2))]
                    metrics.append(dict(k=k, silhouette=float(silhouette_score(z, labels)),
                                        inertia=float(best.inertia_), inertia_ratio_to_k1=float(best.inertia_ / baseline),
                                        calinski_harabasz=float(calinski_harabasz_score(z, labels)),
                                        davies_bouldin=float(davies_bouldin_score(z, labels)),
                                        seed_stability_ari=float(np.mean(stability)),
                                        min_cluster_size=int(np.bincount(labels).min())))
            if not metrics:
                raise ValueError(f'demo_{demo_id}: no valid candidate partitions')
            best_metric = max(metrics, key=lambda m: m['silhouette'])
            k = best_metric['k']
            labels = labels_all[f'labels_k{k}']
            pca = PCA(n_components=2)
            xy = pca.fit_transform(z)
            explained = pca.explained_variance_ratio_
            title = f'demo_{demo_id}: N={n}, K={k}, silhouette={best_metric["silhouette"]:.3f}'
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            for ax in (axes[0, 0], overview_axes[row, 0]):
                ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap='tab10', vmin=0, vmax=9, s=12, alpha=.8)
                ax.set(title=title, xlabel=f'PC1 ({explained[0]:.1%})', ylabel=f'PC2 ({explained[1]:.1%})')
            for ax in (axes[0, 1], overview_axes[row, 1]):
                ax.plot(xy[:, 0], xy[:, 1], color='gray', alpha=.18, linewidth=.6)
                points = ax.scatter(xy[:, 0], xy[:, 1], c=times, cmap='viridis', s=12)
                ax.set(title='Same PCA plane, colored by transition time', xlabel='PC1', ylabel='PC2')
            fig.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Transition t')
            axes[1, 0].scatter(times, labels, c=labels, cmap='tab10', vmin=0, vmax=9, s=10)
            axes[1, 0].set(xlabel='Transition t', ylabel='Cluster ID', yticks=range(k), title='Cluster assignment over time')
            ks = [m['k'] for m in metrics]
            axes[1, 1].plot(ks, [m['silhouette'] for m in metrics], 'o-', label='Silhouette (higher is better)')
            axes[1, 1].axvline(k, color='gray', linestyle='--')
            axes[1, 1].set(xlabel='Number of clusters K', ylabel=f'Full {z.shape[1]}D silhouette', xticks=ks, title='K selection in original latent space')
            axes[1, 1].grid(alpha=.3)
            fig.suptitle(f'{title}; PCA variance retained={explained.sum():.1%}')
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(out / f'demo_{demo_id}.{ext}', dpi=180)
            plt.close(fig)
            columns = 3
            gallery, panels = plt.subplots((len(metrics) + columns - 1) // columns, columns,
                                           figsize=(13, 3.6 * ((len(metrics) + columns - 1) // columns)), squeeze=False)
            for panel, metric in zip(panels.flat, metrics):
                panel.scatter(xy[:, 0], xy[:, 1], c=labels_all[f'labels_k{metric["k"]}'],
                              cmap='tab20', vmin=0, vmax=max(9, args.cluster_max_k - 1), s=9, alpha=.8)
                panel.set(title=f'K={metric["k"]}; silhouette={metric["silhouette"]:.3f}', xlabel='PC1', ylabel='PC2')
            for panel in list(panels.flat)[len(metrics):]:
                panel.set_visible(False)
            gallery.suptitle(f'demo_{demo_id}: candidate partitions on the SAME PCA plane')
            gallery.tight_layout()
            gallery.savefig(out / f'demo_{demo_id}_all_k.png', dpi=160)
            plt.close(gallery)
            np.savez_compressed(out / f'demo_{demo_id}.npz', z_demo=z, timestep=times,
                                observations=states, action_chunks=actions, reconstructed_actions=rebuilt,
                                reconstruction_max_abs=errors, cluster_labels=labels, pca_xy=xy,
                                pca_components=pca.components_, pca_mean=pca.mean_,
                                pca_explained_variance_ratio=explained, **labels_all)
            with (out / f'demo_{demo_id}_metrics.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            record = dict(demo_id=demo_id, num_transitions=lengths[demo_id], num_latents=n,
                          excluded_transition_starts=sorted(set(range(lengths[demo_id])) - set(times.tolist())),
                          selected_k=k, best_silhouette=best_metric['silhouette'],
                          seed_stability_ari=best_metric['seed_stability_ari'],
                          cluster_sizes=np.bincount(labels).tolist(), pca_variance_retained=float(explained.sum()),
                          boundary_fraction=float(np.any(np.abs(actions) >= 1, axis=1).mean()),
                          roundtrip_median=float(np.median(errors)), roundtrip_p95=float(np.quantile(errors, .95)),
                          roundtrip_max=float(errors.max()),
                          roundtrip_within_tolerance_fraction=float((errors <= args.tolerance).mean()),
                          candidates=metrics)
            summary['trajectories'].append(record)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'{title}; sizes={record["cluster_sizes"]}; roundtrip median={record["roundtrip_median"]:.4g}', flush=True)
        overview.tight_layout()
        overview.savefig(out / 'overview.png', dpi=160)
        plt.close(overview)
        lines = ['# Per-trajectory latent clustering', '',
                 f'Checkpoint: `{path.name}`. Seed: {args.seed}. Full raw {summary["latent_dimension"]}D KMeans; PCA for display only.', '',
                 'Reproduce from repository root:', '',
                 '```bash',
                 f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --cluster-trajectories --run-dir {run} --dataset {args.dataset} --steps {step} --demo-ids ' + ' '.join(map(str, ids)) + f' --seed {args.seed} --cluster-max-k {args.cluster_max_k} --batch-size {args.batch_size} --platform {args.platform} --output {args.output}',
                 '```', '',
                 '| Demo | N | Selected K | Silhouette | Seed ARI | PCA variance | Roundtrip median |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for r in summary['trajectories']:
            lines.append(f'| demo_{r["demo_id"]} | {r["num_latents"]} | {r["selected_k"]} | {r["best_silhouette"]:.3f} | {r["seed_stability_ari"]:.3f} | {r["pca_variance_retained"]:.1%} | {r["roundtrip_median"]:.3f} |')
        lines += ['', 'K is exploratory: silhouette ranks K>=2, and does not demonstrate discrete modes versus K=1.',
                  'All valid full action windows are inverted; incomplete terminal windows and windows crossing internal dones are excluded and listed in summary.json.',
                  'Reverse Euler is approximate; clipped actions have no unique inverse. Check roundtrip diagnostics before interpreting clusters.',
                  'Overlapping windows are correlated. Cluster IDs and PCA coordinates are local to each trajectory.',
                  'Each demo NPZ includes all raw z vectors, timesteps, reconstruction errors and labels for every candidate K.', '']
        (out / 'REPORT.md').write_text('\n'.join(lines))
        print(f'Trajectory clustering results: {out}', flush=True)


def analyze_temporal_latent_clouds(source_dir, *, seed=1, permutations=499):
    """Scan continuous windows for whole-cloud localization in the raw z space.

    A directed 10-NN graph excludes temporally overlapping chunks (|dt|<horizon).
    Scan score=(within-window neighbor fraction - window size baseline)/(1-baseline).
    Block-shuffle nulls preserve local sequences; each null repeats the ENTIRE
    scan, correcting selection over starts/lengths. BH adjusts across demos,
    separately for block sizes 4 and 8. This is an exploratory block-exchangeable
    null, not proof of a density mode or a semantic skill.
    """
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances, silhouette_samples
    from scipy.sparse.csgraph import connected_components
    from threadpoolctl import threadpool_limits

    source_dir = Path(source_dir).resolve()
    manifest = json.loads((source_dir / 'summary.json').read_text())
    out = source_dir / 'temporal_clouds'
    out.mkdir(exist_ok=True)
    overview, panels = plt.subplots(len(manifest['trajectories']), 2,
                                   figsize=(12, 3.5 * len(manifest['trajectories'])), squeeze=False)
    results = []
    for row, prior in enumerate(manifest['trajectories']):
        demo_id = prior['demo_id']
        with np.load(source_dir / f'demo_{demo_id}.npz') as data:
            z, times, xy = (data[k] for k in ('z_demo', 'timestep', 'pca_xy'))
            pca_variance = float(data['pca_explained_variance_ratio'].sum())
        n = len(z)
        if not np.all(np.diff(times) == 1):
            raise ValueError('Temporal scan requires contiguous valid timesteps')
        horizon = manifest['horizon']
        with threadpool_limits(limits=1):
            distances = pairwise_distances(z)
        eligible = np.abs(times[:, None] - times[None, :]) >= horizon
        masked = np.where(eligible, distances, np.inf)
        neighbors = min(10, int(eligible.sum(axis=1).min()))
        if neighbors < 1:
            raise ValueError('Not enough nonoverlapping neighbors')
        nn = np.argsort(masked, axis=1)[:, :neighbors]
        graph = np.zeros((n, n), dtype=np.int32)
        graph[np.arange(n)[:, None], nn] = 1
        lengths = np.array([m for m in (12, 20, 32, 48, 64) if m <= n // 2])
        if not len(lengths):
            raise ValueError('Need at least 24 valid z vectors')
        starts = np.concatenate([np.arange(n - m + 1) for m in lengths])
        widths = np.concatenate([np.full(n - m + 1, m) for m in lengths])
        ends = starts + widths
        baseline = (widths - 1) / (n - 1)

        def scan(adjacency):
            integral = np.pad(adjacency.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            edges = integral[ends, ends] - integral[starts, ends] - integral[ends, starts] + integral[starts, starts]
            retention = edges / (widths * neighbors)
            return (retention - baseline) / (1 - baseline), retention

        scores, retention = scan(graph)
        best_idx = int(scores.argmax())
        lo, hi = int(starts[best_idx]), int(ends[best_idx])
        rng = np.random.default_rng(seed + demo_id)
        nulls, p_values = {}, {}
        for block_size in (horizon, 2 * horizon):
            blocks = [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]
            maxima = []
            for repeat in range(permutations):
                order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
                null_score, _ = scan(graph[order][:, order])
                maxima.append(float(null_score.max()))
            nulls[f'block_{block_size}'] = np.asarray(maxima)
            p_values[f'block_{block_size}'] = float((1 + np.sum(np.asarray(maxima) >= scores[best_idx])) / (permutations + 1))
        top_windows, occupied = [], np.zeros(n, dtype=bool)
        global_variance = float(np.mean(np.sum((z - z.mean(0)) ** 2, axis=1)))
        for idx in np.argsort(-scores, kind='stable'):
            a, b = int(starts[idx]), int(ends[idx])
            if occupied[a:b].any():
                continue
            cloud = z[a:b]
            labels = np.zeros(n, dtype=int)
            labels[a:b] = 1
            sil = silhouette_samples(distances, labels, metric='precomputed')
            local_variance = float(np.mean(np.sum((cloud - cloud.mean(0)) ** 2, axis=1)))
            internal = graph[a:b, a:b]
            components, component_labels = connected_components(internal + internal.T, directed=False)
            point_retention = internal.sum(axis=1) / neighbors
            top_windows.append(dict(internal_graph_components=int(components),
                                    largest_component_fraction=float(np.bincount(component_labels).max() / (b-a)),
                                    point_retention_p10=float(np.quantile(point_retention, .1)),
                                    fraction_points_with_majority_neighbors_inside=float((point_retention >= .5).mean()),
                                    start_t=int(times[a]), end_t=int(times[b - 1]), length=b-a,
                                    graph_score=float(scores[idx]), neighbor_retention=float(retention[idx]),
                                    size_baseline=float(baseline[idx]), neighbor_enrichment=float(retention[idx]/baseline[idx]),
                                    variance_ratio_to_whole=local_variance/global_variance,
                                    segment_silhouette=float(sil[a:b].mean()),
                                    positive_silhouette_fraction=float((sil[a:b] > 0).mean())))
            occupied[a:b] = True
            if len(top_windows) == 3:
                break
        # Compactness of the selected whole cloud versus random equal-sized sets.
        m = hi - lo
        random_ratios = []
        for repeat in range(permutations):
            sample = z[rng.choice(n, m, replace=False)]
            random_ratios.append(float(np.mean(np.sum((sample-sample.mean(0))**2, axis=1))/global_variance))
        best = top_windows[0]
        best['random_equal_size_variance_ratio_median'] = float(np.median(random_ratios))
        record = dict(demo_id=demo_id, n=n, nearest_neighbors=neighbors, lengths=lengths.tolist(),
                      best=best, other_nonoverlapping_candidates=top_windows[1:],
                      scan_adjusted_p=p_values, pca_variance_retained=pca_variance)
        results.append(record)
        matrix = np.full((len(lengths), n), np.nan)
        for j, length in enumerate(lengths):
            idx = widths == length
            matrix[j, starts[idx]] = scores[idx]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        for ax in (axes[0, 0], panels[row, 0]):
            ax.imshow(graph, origin='lower', aspect='auto', cmap='Greys', interpolation='nearest', extent=(times[0]-.5, times[-1]+.5, times[0]-.5, times[-1]+.5))
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((times[lo]-.5, times[lo]-.5), m, m, fill=False, edgecolor='red', linewidth=1.5))
            ax.set(title=f'demo_{demo_id}: 10-NN graph in 28D', xlabel='Neighbor transition t', ylabel='Query transition t')
        for ax in (axes[0, 1], panels[row, 1]):
            ax.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=12, alpha=.65, label='Rest of trajectory')
            selected = ax.scatter(xy[lo:hi, 0], xy[lo:hi, 1], c=times[lo:hi], cmap='autumn', s=24, label='Selected interval')
            ax.set(title=f't={times[lo]}..{times[hi-1]}, {m} points; PCA retains {pca_variance:.1%}', xlabel='PC1', ylabel='PC2')
            ax.legend(fontsize=8)
        fig.colorbar(axes[0, 1].collections[-1], ax=axes[0, 1], label='Transition t')
        image = axes[0, 2].imshow(matrix, aspect='auto', origin='lower', cmap='viridis', extent=(times[0]-.5, times[-1]+.5, -.5, len(lengths)-.5))
        axes[0, 2].set(title='Scan over ALL starts and lengths', xlabel='Window start t', ylabel='Window length', yticks=range(len(lengths)), yticklabels=lengths)
        fig.colorbar(image, ax=axes[0, 2], label='Neighborhood localization score')
        local_retention = graph[:, lo:hi].sum(1)/neighbors
        axes[1, 0].plot(times, local_retention, linewidth=.9)
        axes[1, 0].axvspan(times[lo]-.5, times[hi-1]+.5, color='orange', alpha=.2)
        axes[1, 0].axhline(float(baseline[best_idx]), color='gray', linestyle='--', label='Window-size baseline')
        axes[1, 0].set(title="Fraction of each point\'s neighbors in selected interval", xlabel='Query transition t', ylabel='Fraction', ylim=(-.02, 1.02))
        axes[1, 0].legend(fontsize=8)
        for key, values in nulls.items():
            axes[1, 1].hist(values, bins=25, alpha=.5, label=f'{key}, p={p_values[key]:.3f}')
        axes[1, 1].axvline(float(scores[best_idx]), color='red', label='Observed best window')
        axes[1, 1].set(title='Block-shuffle null: maximum over ENTIRE scan', xlabel='Best localization score', ylabel='Count')
        axes[1, 1].legend(fontsize=8)
        axes[1, 2].hist(random_ratios, bins=25, color='gray', alpha=.7)
        axes[1, 2].axvline(best['variance_ratio_to_whole'], color='red', label='Selected interval')
        axes[1, 2].set(title='Whole-cloud spread vs equal-sized random sets', xlabel='Mean squared radius / whole-trajectory variance', ylabel='Count')
        axes[1, 2].legend(fontsize=8)
        fig.suptitle(f'demo_{demo_id}: continuous interval as a spatial cloud; retention={best["neighbor_retention"]:.1%}, segment silhouette={best["segment_silhouette"]:.3f}')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'demo_{demo_id}.{ext}', dpi=170)
        plt.close(fig)
        np.savez_compressed(out / f'demo_{demo_id}_scan.npz', timestep=times, graph=graph, window_starts=starts,
                            window_lengths=widths, scores=scores, neighbor_retention=retention,
                            **{f'null_{key}': values for key, values in nulls.items()})
        with (out / f'demo_{demo_id}_windows.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['start_t', 'end_t', 'length', 'localization_score', 'neighbor_retention', 'size_baseline'])
            writer.writerows(zip(times[starts], times[ends-1], widths, scores, retention, baseline))
        print(f'demo_{demo_id}: t={times[lo]}..{times[hi-1]}, L={m}, retention={best["neighbor_retention"]:.3f}, enrichment={best["neighbor_enrichment"]:.2f}, variance_ratio={best["variance_ratio_to_whole"]:.3f}, silhouette={best["segment_silhouette"]:.3f}, p={p_values}', flush=True)
    for key in results[0]['scan_adjusted_p']:
        ps = np.array([r['scan_adjusted_p'][key] for r in results])
        order = ps.argsort()
        q = np.minimum.accumulate((ps[order]*len(ps)/np.arange(1, len(ps)+1))[::-1])[::-1].clip(0, 1)
        for index, value in zip(order, q):
            results[index].setdefault('bh_q', {})[key] = float(value)
    summary = dict(source_dir=str(source_dir), checkpoint=manifest['checkpoint'], dataset=manifest['dataset'],
                   seed=seed, permutations=permutations, method=analyze_temporal_latent_clouds.__doc__, trajectories=results)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    overview.tight_layout()
    overview.savefig(out / 'overview.png', dpi=160)
    plt.close(overview)
    lines = ['# 连续时间片段是否在 z 空间中成团', '',
             '复用同一 Square checkpoint、同一 demo 数据集和上次保存的全部 28 维 z；无重新反解或修改 z。', '',
             '每条轨迹扫描长度 12/20/32/48/64 的所有连续窗口。构建原始 28 维 10-NN 有向图，排除时间距离小于 action horizon=4 的重叠窗口邻居。',
             '保留率=片段内每个点的 10 个邻居中也落在片段内的平均比例；富集倍数=保留率/片段大小基线。整组方差比=片段到自身中心的平均平方半径/整条轨迹对应量，越小越紧。',
             '片段轮廓系数将该段全部点作为一组、其余点作为另一组，报告该段点的平均 silhouette；接近 0 表示分离弱，负数提示该段不如其余点构成清晰一簇。',
             f'每条轨迹分别做 {permutations} 次 block=4 和 block=8 的时间分块置换，每次重新扫描所有窗口并取最大分数，避免挑最好窗口造成的偏差；再对 8 条轨迹做 BH 校正。',
             '分块置换依赖块可交换的探索性假设；随机同大小点集的紧致度直方图仅作描述，不当作独立显著性检验。没有要求每一个点都接近，也不证明密度模式或语义技能。', '',
             '| Demo | 最佳连续区间（含端点） | 步数 | 邻居保留率 | 富集倍数 | 整组方差比 | 片段 silhouette | q(block4) | q(block8) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        b = r['best']
        qs = list(r['bh_q'].values())
        lines.append(f'| demo_{r["demo_id"]} | {b["start_t"]}–{b["end_t"]} | {b["length"]} | {b["neighbor_retention"]:.1%} | {b["neighbor_enrichment"]:.2f} | {b["variance_ratio_to_whole"]:.3f} | {b["segment_silhouette"]:.3f} | {qs[0]:.3f} | {qs[1]:.3f} |')
    lines += ['', '图示：邻域图的红框表示该连续片段；空间图中灰色是其余 z，彩色是该段全部 z；其他面板给出全窗口扫描、时序邻域保留率、置换对照和整组紧致度。PCA 仅用于显示，所有统计都在原始 28 维空间计算。',
              '原始反解的 reverse Euler 数值误差、最终 action clipping 不可逆及末尾无有效完整窗口的限制仍然适用。', '',
              '复现：', '```bash', f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --temporal-clusters-from {source_dir} --temporal-permutations {permutations} --seed {seed}', '```', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(f'Temporal cloud results: {out}', flush=True)


def analyze_all_demo_latent_clusters(args, run, config, paths):
    """Joint full-data KMeans in raw z, with sampled evaluation/density controls.

    Inverts ALL valid chunks from ALL demos, regardless of --max-samples or
    --demo-ids. Cache reuse checks checkpoint SHA256, data SHA256 and horizon.
    PCA uses all z. UMAP/HDBSCAN use an explicitly saved uniform sample only.
    Gaussian-reference KMeans is fitted on equal-size real/reference samples,
    for descriptive calibration, not a significance test or a mode-count proof.
    """
    import csv
    import h5py
    import jax
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import (silhouette_score, pairwise_distances,
                                 adjusted_rand_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    from threadpoolctl import threadpool_limits

    def file_digest(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    horizon = config['horizon_length'] if config['action_chunking'] else 1
    dataset_sha = file_digest(args.dataset)
    data, available = load_demos(args.dataset, horizon, 0, args.seed)
    n, dimension = data['action_chunks'].shape
    with h5py.File(args.dataset, 'r') as f:
        demo_lengths = {int(name[5:]): len(f['data'][name]['actions']) for name in f['data']}
    if set(np.unique(data['demo_id'])) != set(demo_lengths):
        raise ValueError('Some demo has no valid full chunks')
    if not all(np.isfinite(data[key]).all() for key in ('observations', 'action_chunks')):
        raise ValueError('Nonfinite demo observations/actions')
    if np.max(np.abs(data['action_chunks'])) > 1:
        raise ValueError('Demo actions outside [-1,1]')
    print(f'ALL Square demos: {len(demo_lengths)}; transitions={sum(demo_lengths.values())}; valid full chunks={n}; z dimension={dimension}', flush=True)
    invert, _ = make_functions(config, dimension, args)
    def analyze_cloud(z, data, errors, out, source_indices, scope):
        n = len(z)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        eval_idx = np.sort(rng.choice(n, min(n, args.global_eval_samples), replace=False))
        plot_idx = np.sort(rng.choice(n, min(n, 12000), replace=False))
        eval_z = z[eval_idx]
        covariance = np.cov(z, rowvar=False)
        gaussian = rng.multivariate_normal(z.mean(0), covariance, size=len(eval_idx)).astype(np.float32)
        metrics, all_labels, centers = [], {}, {}
        with threadpool_limits(limits=4):
            eval_distances = pairwise_distances(eval_z)
            gaussian_distances = pairwise_distances(gaussian)
            # Query-to-ALL distances keep singleton/extreme clusters represented
            # even when no point from a tiny cluster enters the query sample.
            reference_distances = pairwise_distances(eval_z, z)
            from scipy.sparse import csr_matrix

            def population_silhouette(labels):
                cluster_counts = np.bincount(labels)
                membership = csr_matrix((np.ones(n), (np.arange(n), labels)),
                                        shape=(n, len(cluster_counts)))
                sums = (membership.T @ reference_distances.T).T
                query_labels = labels[eval_idx]
                intra = sums[np.arange(len(eval_idx)), query_labels] / np.maximum(cluster_counts[query_labels]-1, 1)
                means = sums / cluster_counts[None, :]
                means[np.arange(len(eval_idx)), query_labels] = np.inf
                inter = means.min(axis=1)
                values = np.divide(inter-intra, np.maximum(intra, inter), out=np.zeros_like(intra), where=np.maximum(intra, inter)>0)
                values[cluster_counts[query_labels]==1] = 0
                return float(values.mean())
            baseline_inertia = float(np.sum((z.astype(np.float64)-z.mean(0, dtype=np.float64))**2))
            for k in range(2, args.global_max_k + 1):
                fits = [KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=args.seed+i).fit(z) for i in range(2)]
                best = min(fits, key=lambda f: f.inertia_)
                labels = best.labels_
                all_labels[f'labels_k{k}'] = labels
                centers[f'centers_k{k}'] = best.cluster_centers_
                sizes = np.bincount(labels, minlength=k)
                sample_fit = KMeans(n_clusters=k, n_init=5, random_state=args.seed).fit(eval_z)
                null_fit = KMeans(n_clusters=k, n_init=5, random_state=args.seed).fit(gaussian)
                exact_inertia = float(np.sum((z.astype(np.float64)-best.cluster_centers_.astype(np.float64)[labels])**2))
                metric = dict(k=k, silhouette=population_silhouette(labels),
                              inertia=exact_inertia, sklearn_float32_inertia=float(best.inertia_),
                              inertia_ratio_to_k1=exact_inertia/baseline_inertia,
                              calinski_harabasz=float(calinski_harabasz_score(eval_z, labels[eval_idx])) if len(np.unique(labels[eval_idx]))>1 else None,
                              davies_bouldin=float(davies_bouldin_score(eval_z, labels[eval_idx])) if len(np.unique(labels[eval_idx]))>1 else None,
                              seed_stability_ari=float(adjusted_rand_score(fits[0].labels_, fits[1].labels_)),
                              min_cluster_size=int(sizes.min()), max_cluster_size=int(sizes.max()),
                              sample_refit_silhouette=float(silhouette_score(eval_distances, sample_fit.labels_, metric='precomputed')),
                              gaussian_reference_silhouette=float(silhouette_score(gaussian_distances, null_fit.labels_, metric='precomputed')))
                metrics.append(metric)
                with (out / 'kmeans_metrics.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
                    writer.writeheader()
                    writer.writerows(metrics)
                np.savez_compressed(out / 'kmeans_partitions.npz', **all_labels, **centers, eval_indices=eval_idx, source_row_indices=source_indices)
                print(f'Global K={k}: silhouette={metric["silhouette"]:.4f}; ARI={metric["seed_stability_ari"]:.3f}; Gaussian reference={metric["gaussian_reference_silhouette"]:.4f}', flush=True)
        selected = max(metrics, key=lambda m: m['silhouette'])
        best_k = selected['k']
        labels = all_labels[f'labels_k{best_k}']
        with threadpool_limits(limits=4):
            pca = PCA(n_components=4).fit(z)
            xy = pca.transform(z)
        counts = np.zeros((len(demo_lengths), best_k), dtype=np.int32)
        demo_ids = np.array(sorted(demo_lengths))
        for i, demo in enumerate(demo_ids):
            counts[i] = np.bincount(labels[data['demo_id']==demo], minlength=best_k)
        density_metrics, density_labels = [], {}
        with threadpool_limits(limits=4):
            for min_size, min_samples in ((100, 10), (100, 30), (300, 10), (300, 30)):
                density = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples, n_jobs=4).fit(z[plot_idx])
                dl = density.labels_
                key = f'minsize{min_size}_minsamples{min_samples}'
                density_labels[key] = dl
                sizes = {int(label): int((dl==label).sum()) for label in np.unique(dl)}
                item = dict(min_cluster_size=min_size, min_samples=min_samples, clusters=sum(label>=0 for label in sizes),
                            noise_fraction=float((dl<0).mean()), label_sizes=sizes)
                density_metrics.append(item)
                print(f'HDBSCAN sample N={len(plot_idx)}, {key}: clusters={item["clusters"]}, noise={item["noise_fraction"]:.1%}', flush=True)
        print('Computing sampled UMAP visualization (not used for clustering)', flush=True)
        import umap
        with threadpool_limits(limits=4):
            embedding = umap.UMAP(n_neighbors=30, min_dist=.15, n_components=2,
                                  random_state=args.seed, n_jobs=1).fit_transform(z[plot_idx])
        np.savez_compressed(out / 'projections_and_density.npz', pca_xy=xy, pca_components=pca.components_,
                            pca_mean=pca.mean_, pca_variance_ratio=pca.explained_variance_ratio_,
                            plot_indices=plot_idx, eval_indices=eval_idx, umap_xy=embedding,
                            demo_ids=demo_ids, demo_cluster_counts=counts, **density_labels)
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=labels, cmap='tab20', s=1, alpha=.3, rasterized=True)
        axes[0, 0].set(title=f'ALL {n} z: global KMeans K={best_k}', xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0, 1].hexbin(xy[:, 0], xy[:, 1], gridsize=80, bins='log', mincnt=1, cmap='magma')
        axes[0, 1].set(title='ALL z: PCA density (log counts)', xlabel='PC1', ylabel='PC2')
        scatter = axes[0, 2].scatter(xy[plot_idx, 0], xy[plot_idx, 1], c=data['demo_id'][plot_idx], cmap='turbo', s=3, alpha=.5)
        fig.colorbar(scatter, ax=axes[0, 2], label='Demo ID (categorical IDs)')
        axes[0, 2].set(title='Uniform sample: demo source', xlabel='PC1', ylabel='PC2')
        axes[1, 0].scatter(embedding[:, 0], embedding[:, 1], c=labels[plot_idx], cmap='tab20', s=3, alpha=.5)
        axes[1, 0].set(title=f'UMAP sample N={len(plot_idx)}; global KMeans colors', xlabel='UMAP1', ylabel='UMAP2')
        ks = [m['k'] for m in metrics]
        axes[1, 1].plot(ks, [m['silhouette'] for m in metrics], 'o-', label='Full-data fit; sample evaluation')
        axes[1, 1].plot(ks, [m['sample_refit_silhouette'] for m in metrics], '--', label='Real sample refit')
        axes[1, 1].plot(ks, [m['gaussian_reference_silhouette'] for m in metrics], '--', label='Matched single Gaussian sample')
        axes[1, 1].axvline(best_k, color='gray', linestyle=':')
        axes[1, 1].set(title='Silhouette in original 28D', xlabel='K', ylabel='Silhouette')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=.3)
        axes[1, 2].imshow(counts/counts.sum(1, keepdims=True), aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[1, 2].set(title='Cluster fraction in each of 300 demos', xlabel='Cluster ID', ylabel='Demo index', xticks=range(best_k))
        fig.suptitle(f'Square: {len(demo_lengths)} demos, {n} raw 28D z jointly clustered; best silhouette={selected["silhouette"]:.4f}')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'overview.{ext}', dpi=180)
        plt.close(fig)
        gallery, panels = plt.subplots(2, 3, figsize=(16, 10))
        candidates = sorted(set([2, 3, 4, 6, 10, best_k]))
        for panel, k in zip(panels.flat, candidates):
            if f'labels_k{k}' not in all_labels:
                panel.set_visible(False)
                continue
            panel.scatter(xy[plot_idx, 0], xy[plot_idx, 1], c=all_labels[f'labels_k{k}'][plot_idx], cmap='tab20', s=2, alpha=.4)
            panel.set(title=f'K={k}; silhouette={next(m["silhouette"] for m in metrics if m["k"]==k):.4f}', xlabel='PC1', ylabel='PC2')
        for panel in list(panels.flat)[len(candidates):]:
            panel.set_visible(False)
        gallery.suptitle('Global candidate partitions on the SAME PCA plane (display sample only)')
        gallery.tight_layout()
        gallery.savefig(out / 'candidate_k.png', dpi=180)
        plt.close(gallery)
        density_fig, panels = plt.subplots(2, 2, figsize=(13, 11))
        for panel, (key, dl), metric in zip(panels.flat, density_labels.items(), density_metrics):
            noise = dl<0
            panel.scatter(embedding[noise, 0], embedding[noise, 1], c='lightgray', s=2, alpha=.4)
            panel.scatter(embedding[~noise, 0], embedding[~noise, 1], c=dl[~noise], cmap='tab20', s=4, alpha=.6)
            panel.set(title=f'{key}: {metric["clusters"]} clusters, {metric["noise_fraction"]:.1%} noise', xlabel='UMAP1 (display only)', ylabel='UMAP2')
        density_fig.suptitle('HDBSCAN fitted in ORIGINAL 28D on a fixed 12,000-point sample')
        density_fig.tight_layout()
        density_fig.savefig(out / 'density_check.png', dpi=180)
        plt.close(density_fig)
        summary = dict(checkpoint=str(path), dataset=str(args.dataset.resolve()), **identity,
                       num_demos=len(demo_lengths), num_transitions=sum(demo_lengths.values()), num_latents=n,
                       latent_dimension=dimension, seed=args.seed, all_demos_used=True, all_valid_windows_used=(scope=='all_raw_z'), analysis_scope=scope,
                       excluded_extreme_count=int(available-n), extreme_norm_threshold=float(extreme_threshold),
                       per_demo=[dict(demo_id=demo, transitions=length, valid_z=int((data['demo_id']==demo).sum())) for demo, length in sorted(demo_lengths.items())],
                       fitting='KMeans on ALL raw 28D z without standardization or PCA; 2 seeds x 5 restarts per K',
                       evaluation=f'Fixed uniform {len(eval_idx)} query points; silhouette uses distances to ALL reference z, including tiny clusters',
                       selected_k=best_k, best_silhouette=selected['silhouette'], metrics=metrics,
                       cluster_sizes=np.bincount(labels).tolist(), demos_per_cluster=(counts>0).sum(0).tolist(),
                       pca_variance_ratio=pca.explained_variance_ratio_.tolist(),
                       density_sample_size=len(plot_idx), density_metrics=density_metrics,
                       roundtrip_median=float(np.median(errors)), roundtrip_p95=float(np.quantile(errors,.95)),
                       roundtrip_max=float(errors.max()), boundary_fraction=float(np.any(np.abs(data['action_chunks'])>=1, axis=1).mean()))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        lines = ['# 全部 Square demo 的联合 latent 聚类', '',
                 f'Checkpoint: `{path}`。数据集: `{args.dataset}`。',
                 f'分析范围: {scope}。{len(demo_lengths)} 条轨迹、{sum(demo_lengths.values())} 个原始 transition，本次拟合 {n} 个 {dimension} 维 z。全量反解共 {available} 个。',
                 f'极端点敏感性规则：模长 > 10 × 全量模长中位数（阈值 {extreme_threshold:.6g}）；本范围排除 {available-n} 个点。原始全量结果不排除任何点。',
                 '原始反解按 loader 排除跨 done 和不足 horizon 的窗口；没有随机抽样再反解。所有 z 均为有限值。主体敏感性分析仅在全量反解后使用上述规则过滤。',
                 f'KMeans 使用全部 {n} 个原始 z，尝试 K=2～{args.global_max_k}；每个 K 两组随机种子，各 n_init=5。',
                 f'轮廓系数对固定均匀抽样的 {len(eval_idx)} 个查询点计算到全部参考 z 的距离，保留极小簇的影响；查询均值估计全局得分。最佳候选 K={best_k}，silhouette={selected["silhouette"]:.4f}。',
                 '协方差匹配的单峰高斯对照与真实样本使用相同样本量、相同 KMeans 设置重新拟合。该对照为描述性参照，没有给出显著性检验，也不能据此直接确定模态数。',
                 f'HDBSCAN 在固定 {len(plot_idx)} 点样本的原始 28 维空间拟合，扫描 4 组参数；不代表全部数据的密度聚类标签。UMAP 也只使用该样本，二维岛屿不可直接作为存在子簇的证据。',
                 '全局 PCA 使用全部 z 拟合。长轨迹按 transition 数量贡献更多权重。全局簇标签在所有轨迹间统一；demo_cluster_counts 可查看跨轨迹共享程度。', '',
                 '| K | silhouette | 种子 ARI | 惯量/K=1 | 真实样本重拟合 silhouette | 单峰高斯对照 silhouette |',
                 '|---:|---:|---:|---:|---:|---:|']
        for m in metrics:
            lines.append(f'| {m["k"]} | {m["silhouette"]:.4f} | {m["seed_stability_ari"]:.3f} | {m["inertia_ratio_to_k1"]:.3f} | {m["sample_refit_silhouette"]:.4f} | {m["gaussian_reference_silhouette"]:.4f} |')
        lines += ['', '密度聚类补充：', '```json', json.dumps(density_metrics, indent=2), '```', '',
                  f'反解再解码的逐窗口最大绝对误差：median={summary["roundtrip_median"]:.4f}, p95={summary["roundtrip_p95"]:.4f}, max={summary["roundtrip_max"]:.4f}。最终 action clipping 不可逆，reverse Euler 是近似反解。', '',
                  '文件：上层原始结果目录 all_latents.npz 含全部 z/源数据与重建；本目录 kmeans_partitions.npz 含所有 K 的本范围标签、中心及 source_row_indices（对应原始全量缓存行号）；projections_and_density.npz 含投影、抽样索引、密度标签和 demo×cluster 计数。', '',
                  '复现（已有 SHA256 匹配缓存会复用反解结果）：', '```bash',
                  f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --cluster-all-trajectories --steps {step} --global-max-k {args.global_max_k} --global-eval-samples {args.global_eval_samples} --batch-size {args.batch_size} --platform {args.platform} --seed {args.seed}', '```', '']
        (out / 'REPORT.md').write_text('\n'.join(lines))
        print(f'All-demo clustering results: {out}', flush=True)

    for path in (paths if args.steps else paths[-1:]):
        step = int(path.stem.split('_')[-1])
        out = args.output / run.name / f'all_demo_clusters_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = file_digest(path)
        identity = dict(checkpoint_sha256=checkpoint_sha, dataset_sha256=dataset_sha,
                        horizon=horizon, inverse_method='reverse_time_explicit_euler', flow_steps=config['flow_steps'],
                        matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default'), platform=args.platform)
        cache = out / 'all_latents.npz'
        identity_path = out / 'inversion_identity.json'
        if cache.exists() and identity_path.exists() and json.loads(identity_path.read_text()) == identity:
            with np.load(cache) as saved:
                z, errors, rebuilt = (saved[k] for k in ('z_demo', 'reconstruction_max_abs', 'reconstructed_actions'))
                np.testing.assert_array_equal(saved['demo_id'], data['demo_id'])
                np.testing.assert_array_equal(saved['timestep'], data['timestep'])
            print('Reused SHA256-validated all-demo inversion cache', flush=True)
        else:
            with path.open('rb') as f:
                payload = pickle.load(f)
            params = payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow']
            del payload
            z = np.empty((n, dimension), dtype=np.float32)
            rebuilt = np.empty_like(z)
            errors = np.empty(n, dtype=np.float32)
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                pad = args.batch_size - (end - start)
                states = np.pad(data['observations'][start:end], ((0, pad), (0, 0)), mode='edge')
                actions = np.pad(data['action_chunks'][start:end], ((0, pad), (0, 0)), mode='edge')
                result = invert(params, states, actions)
                for target, values in zip((z, rebuilt, errors), result):
                    target[start:end] = np.asarray(values)[:end-start]
                if start == 0 or end == n or (start // args.batch_size) % 10 == 0:
                    print(f'Inverted {end}/{n}', flush=True)
            np.savez_compressed(cache, z_demo=z, reconstructed_actions=rebuilt, reconstruction_max_abs=errors,
                                **data)
            identity_path.write_text(json.dumps(identity, indent=2)+'\n')
            del params
        if z.shape != (n, dimension) or not np.isfinite(z).all():
            raise ValueError('Invalid full inversion cache')
        norms = np.linalg.norm(z, axis=1)
        extreme_threshold = float(10 * np.median(norms))
        core = norms <= extreme_threshold
        plot_all_demo_latent_norms(out)
        print(f'Extreme norm threshold={extreme_threshold:.6g}; excluded in sensitivity only: {(~core).sum()}/{len(z)}', flush=True)
        analyze_cloud(z, data, errors, out, np.arange(len(z)), 'all_raw_z')
        analyze_cloud(z[core], {key: values[core] for key, values in data.items()}, errors[core],
                      out / 'typical_scale_sensitivity', np.flatnonzero(core), 'typical_scale_sensitivity')


def plot_all_demo_latent_norms(output_dir):
    """Document every excluded extreme row; never remove it from raw results."""
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    with np.load(out / 'all_latents.npz') as data:
        norms = np.linalg.norm(data['z_demo'], axis=1)
        errors = data['reconstruction_max_abs']
        demo_ids, times = data['demo_id'], data['timestep']
    threshold = float(10*np.median(norms))
    extreme = norms > threshold
    indices = np.flatnonzero(extreme)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.log10(norms), bins=100, color='steelblue', log=True)
    axes[0].axvline(np.log10(threshold), color='red', linestyle='--', label='10 x median norm')
    axes[0].set(title=f'ALL {len(norms)} latent norms; {extreme.sum()} extreme rows', xlabel='log10(||z||)', ylabel='Count (log scale)')
    axes[0].legend()
    axes[1].scatter(np.arange(len(norms)), norms, s=1, alpha=.3)
    axes[1].scatter(indices, norms[indices], color='red', s=15)
    axes[1].axhline(threshold, color='red', linestyle='--')
    axes[1].set(yscale='log', title='Latent magnitude by source row (demo/time order)', xlabel='Global source row', ylabel='||z|| (log scale)')
    fig.tight_layout()
    fig.savefig(out / 'norm_diagnostics.png', dpi=180)
    plt.close(fig)
    with (out / 'extreme_rows.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['source_row', 'demo_id', 'timestep', 'latent_norm', 'roundtrip_max_abs'])
        writer.writerows((int(i), int(demo_ids[i]), int(times[i]), float(norms[i]), float(errors[i])) for i in indices)
    diagnostics = dict(norm_quantiles={str(q):float(np.quantile(norms,q)) for q in (0,.5,.9,.99,.999,1)},
                       rule='latent norm > 10 * global median norm', threshold=threshold,
                       extreme_count=int(extreme.sum()), retained_count=int((~extreme).sum()),
                       extreme_demo_ids=np.unique(demo_ids[extreme]).tolist())
    (out / 'norm_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    return diagnostics


def main():
    args = parse_args()
    if args.temporal_clusters_from:
        analyze_temporal_latent_clouds(args.temporal_clusters_from, seed=args.seed, permutations=args.temporal_permutations)
        return
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
    if args.cluster_all_trajectories:
        analyze_all_demo_latent_clusters(args, run, config, paths)
        return
    if args.cluster_trajectories:
        analyze_trajectory_latent_clusters(args, run, config, paths)
        return
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
