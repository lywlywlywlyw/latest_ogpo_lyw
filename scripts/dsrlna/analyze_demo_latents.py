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
All inversion entry points use the SAME velocity field and query grid:
  x = demo_action
  for i = flow_steps-1, ..., 0:
      x = x - v(s,x,i/flow_steps) / flow_steps
  z_demo = x
The state s stays fixed. No clipping, optimization, or random initialization is
applied during inversion. This explicit subtraction on the reversed forward
query grid is NOT an exact inverse of finite-step forward Euler. Final
clipping is non-injective: at a boundary we start at the observed action itself
and cannot recover its unknown pre-clipping value. We report boundary_action
and forward round-trip errors without using them to optimize z_demo.
The shared implementation is ogpo/agents/modules/flow_inverse.py.

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

Invert ALL demos at shifted reverse times, without clustering:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --invert-all-shifted-times --steps 2000000 --batch-size 256
All modes query 0.9,...,0 for flow_steps=10. This mode keeps the original forward
decoder for reconstruction diagnostics, and writes a separate output folder.

Fixed-state single-coordinate experiment:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --latent-coordinate-experiment --steps 2000000 --batch-size 128
Writes requested expert-relative effects, baseline-corrected effects, signed
response curves, clipped/preclip Jacobians, and per-state coordinate metrics.

Experiment 4 (Square): activity, block sensitivities, SVD and finite directions:
  .venv/bin/python scripts/dsrlna/analyze_demo_latents.py --flow-geometry-experiment --steps 2000000 --batch-size 256
Samples 25 trajectories x 4 relative-progress bins, prioritizes preclip
geometry, and includes conditional-spread controls and trajectory bootstrap.
--geometry-postprocess-from reuses saved model outputs without reinversion.

"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ogpo.agents.modules.flow_inverse import (
    FLOW_INVERSE_METHOD, FLOW_INVERSE_OUTPUT_DIR,
    inverse_time_grid, invert_flow_actions_ode,
)


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
    p.add_argument('--invert-all-shifted-times', action='store_true', help='Invert all valid Square chunks with the shared inverse, with norm/error diagnostics only; no clustering')
    p.add_argument('--latent-coordinate-experiment', action='store_true', help='One-coordinate latent interventions at fixed offline states')
    p.add_argument('--perturb-demo-id', type=int, default=226)
    p.add_argument('--perturb-timestep', type=int, default=70)
    p.add_argument('--perturb-num-states', type=int, default=8, help='Primary fixed state plus seeded states from other demos')
    p.add_argument('--perturb-deltas', type=float, nargs='+', default=[-.5, -.25, -.1, -.05, .05, .1, .25, .5])
    p.add_argument('--jacobian-minus-identity-from', type=Path, help='Small postprocessing-only J-I diagnostic from a saved latent-coordinate experiment')
    p.add_argument('--latent-action-distance-experiment', action='store_true', help='Fixed-state latent/action distance relations around inverted demos')
    p.add_argument('--distance-num-states', type=int, default=100)
    p.add_argument('--distance-num-directions', type=int, default=20, help='Same random unit directions reused across radii and states')
    p.add_argument('--distance-radii', type=float, nargs='+', default=[.1,.25,.5,1.,2.])
    p.add_argument('--flow-geometry-experiment', action='store_true', help='Experiment 4: activity, component sensitivities, SVD and finite perturbations')
    p.add_argument('--geometry-postprocess-from', type=Path, help='Regenerate experiment 4 statistics and plots from saved arrays; no model execution')
    p.add_argument('--geometry-num-trajectories', type=int, default=25)
    p.add_argument('--geometry-phase-bins', type=int, default=4)
    p.add_argument('--geometry-prior-samples', type=int, default=128)
    p.add_argument('--geometry-bootstrap', type=int, default=2000)
    p.add_argument('--geometry-model-only', action='store_true', help='Save model outputs; statistics can be resumed with --geometry-postprocess-from')
    args = p.parse_args()
    if min(args.geometry_num_trajectories, args.geometry_phase_bins) < 1 or args.geometry_prior_samples < 2 or args.geometry_bootstrap < 100:
        p.error('geometry trajectory/phase counts must be positive; prior-samples >= 2 and bootstrap >= 100')
    if args.distance_num_states < 1 or args.distance_num_directions < 2:
        p.error('distance-num-states >= 1 and distance-num-directions >= 2 required')
    if len(set(args.distance_radii)) < 2 or any(not math.isfinite(r) or r <= 0 for r in args.distance_radii):
        p.error('At least two distinct finite positive distance-radii required')
    if args.perturb_num_states < 1 or any(not math.isfinite(d) or d == 0 for d in args.perturb_deltas):
        p.error('perturb-num-states >= 1 and finite nonzero perturb-deltas required')
    if args.global_max_k < 2 or args.global_eval_samples < 100:
        p.error('global-max-k >= 2 and global-eval-samples >= 100 required')
    if args.temporal_permutations < 19:
        p.error('temporal-permutations must be >= 19')
    if sum((args.relationship_only, args.cluster_trajectories, args.cluster_all_trajectories, args.invert_all_shifted_times, args.latent_coordinate_experiment, args.latent_action_distance_experiment, args.flow_geometry_experiment, bool(args.geometry_postprocess_from), bool(args.jacobian_minus_identity_from), bool(args.temporal_clusters_from))) > 1:
        p.error('Select only one analysis mode')
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
        latent_provenance='explicit subtraction; decoder final clipping and discretization affect z_demo',
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


def make_functions(config, dimension, args, return_decoder=False):
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

    def forward(params, s, z, clip_final=True):
        return sample_flow_actions_ode(
            lambda obs, x, t, is_encoded: velocity(params, obs, x, t), s, None,
            steps, dimension, act_min=-1. if clip_final else -jnp.inf,
            act_max=1. if clip_final else jnp.inf, clip_intermediate=False,
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

    if return_decoder:
        decode = jax.jit(lambda params, states, z: forward(params, states, z))
        decode_raw = jax.jit(lambda params, states, z: forward(params, states, z, clip_final=False))
        return invert, distribution, decode, decode_raw
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
        out = args.output / run.name / FLOW_INVERSE_OUTPUT_DIR / f'trajectory_clusters_step_{step}_seed_{args.seed}'
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
                       inverse_method=FLOW_INVERSE_METHOD,
                       reverse_query_times=list(inverse_time_grid(config['flow_steps'])),
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
        out = args.output / run.name / FLOW_INVERSE_OUTPUT_DIR / f'all_demo_clusters_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = file_digest(path)
        identity = dict(checkpoint_sha256=checkpoint_sha, dataset_sha256=dataset_sha,
                        horizon=horizon, inverse_method=FLOW_INVERSE_METHOD,
                        reverse_query_times=list(inverse_time_grid(config['flow_steps'])), flow_steps=config['flow_steps'],
                        matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default'), platform=args.platform)
        cache = out / 'all_latents.npz'
        identity_path = out / 'inversion_identity.json'
        # The inversion-only experiment already covers every valid chunk. Reuse
        # it only after checking weights, dataset, time grid, and exact inputs.
        candidate_dir = args.output / run.name / f'all_demo_inverse_step_{step}_time_shift_1'
        candidate = candidate_dir / 'all_latents.npz'
        candidate_manifest = candidate_dir / 'summary.json'
        if not cache.exists() and candidate.exists() and candidate_manifest.exists():
            source = json.loads(candidate_manifest.read_text())
            compatible = (source.get('checkpoint_sha256') == checkpoint_sha
                          and source.get('dataset_sha256') == dataset_sha
                          and source.get('reverse_query_times') == list(inverse_time_grid(config['flow_steps']))
                          and source.get('flow_steps') == config['flow_steps']
                          and source.get('horizon') == horizon
                          and source.get('platform') == args.platform
                          and source.get('batch_size') == args.batch_size
                          and source.get('no_latent_filtering') is True)
            if compatible:
                with np.load(candidate) as saved:
                    for key in data:
                        np.testing.assert_array_equal(saved[key], data[key])
                    imported = {key: saved[key] for key in
                                ('z_demo', 'reconstructed_actions', 'reconstruction_max_abs')}
                if imported['z_demo'].shape != (n, dimension) or not np.isfinite(imported['z_demo']).all():
                    raise ValueError('Invalid inversion-only source cache')
                np.savez_compressed(cache, **data, **imported)
                identity_path.write_text(json.dumps(identity, indent=2) + '\n')
                (out / 'inversion_source.json').write_text(json.dumps(dict(
                    source=str(candidate), verified='checkpoint/dataset SHA256, grid, platform, batch size, all input arrays',
                    num_demos=len(demo_lengths), num_latents=n), indent=2) + '\n')
                print(f'Reused ALL {n} corrected-grid inversions from {candidate}', flush=True)
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
        if not core.all():
            analyze_cloud(z[core], {key: values[core] for key, values in data.items()}, errors[core],
                          out / 'typical_scale_sensitivity', np.flatnonzero(core), 'typical_scale_sensitivity')
        else:
            (out / 'sensitivity_not_needed.json').write_text(json.dumps(dict(
                reason='No extreme inversions: the sensitivity subset is identical to the full dataset',
                num_latents=n, excluded_count=0, threshold=extreme_threshold), indent=2) + '\n')
            print('No extreme z; sensitivity subset is identical to ALL data, so no duplicate clustering.', flush=True)


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


def analyze_all_demo_shifted_inverse(args, run, config, paths):
    """All-demo shared-grid inversion; no filtering or clustering.

    Query times are (K-1)/K,...,0; K updates of -v/K. State stays fixed.
    Forward reconstruction always uses the unchanged decoder at 0,...,(K-1)/K.
    Uses the same shared inverse as every other current entry point.
    Existing historical artifacts retain their original recorded values.
    """
    import csv
    import h5py
    import jax
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def statistics(z, error):
        norm = np.linalg.norm(z.astype(np.float64), axis=1)
        finite = np.isfinite(z).all(1) & np.isfinite(norm)
        valid_norm = norm[finite]
        finite_error = error[np.isfinite(error)]
        median = float(np.median(valid_norm)) if len(valid_norm) else None
        stats = dict(num_samples=len(z), finite_count=int(finite.sum()), nonfinite_count=int((~finite).sum()),
                     norm_quantiles={str(q):float(np.quantile(valid_norm,q)) if len(valid_norm) else None for q in (.5,.9,.99,.999,1.)},
                     norm_counts_above={str(t):int((norm>t).sum()) for t in (5,10,20,50,100,1000,10000,1000000)},
                     ten_median_threshold=10*median if median is not None else None,
                     count_above_ten_median=int((norm>10*median).sum()) if median is not None else None,
                     roundtrip_nonfinite_count=int((~np.isfinite(error)).sum()),
                     roundtrip_median=float(np.median(finite_error)) if len(finite_error) else None,
                     roundtrip_p95=float(np.quantile(finite_error,.95)) if len(finite_error) else None,
                     roundtrip_max=float(np.max(finite_error)) if len(finite_error) else None)
        return norm, stats

    horizon = config['horizon_length'] if config['action_chunking'] else 1
    data, available = load_demos(args.dataset, horizon, 0, args.seed)
    n, dimension = data['action_chunks'].shape
    with h5py.File(args.dataset,'r') as f:
        lengths={int(name[5:]):len(f['data'][name]['actions']) for name in f['data']}
    if set(np.unique(data['demo_id'])) != set(lengths):
        raise ValueError('Some demo has no valid chunks')
    if not all(np.isfinite(data[key]).all() for key in ('observations','action_chunks')):
        raise ValueError('Nonfinite demo inputs')
    if np.max(np.abs(data['action_chunks']))>1:
        raise ValueError('Demo action outside [-1,1]')
    dataset_sha=sha256_file(args.dataset)
    reverse_times=list(inverse_time_grid(config['flow_steps']))
    invert,_=make_functions(config,dimension,args)
    print(f'Square ALL demos={len(lengths)}, transitions={sum(lengths.values())}, valid chunks={n}, dimension={dimension}, reverse query times={reverse_times}',flush=True)
    for path in (paths if args.steps else paths[-1:]):
        step=int(path.stem.split('_')[-1])
        out=args.output/run.name/f'all_demo_inverse_step_{step}_time_shift_1'
        out.mkdir(parents=True,exist_ok=True)
        checkpoint_sha=sha256_file(path)
        with path.open('rb') as f:payload=pickle.load(f)
        params=payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow']
        del payload
        z=np.empty((n,dimension),dtype=np.float32)
        rebuilt=np.empty_like(z)
        errors=np.empty(n,dtype=np.float32)
        for start in range(0,n,args.batch_size):
            end=min(start+args.batch_size,n);pad=args.batch_size-(end-start)
            states=np.pad(data['observations'][start:end],((0,pad),(0,0)),mode='edge')
            actions=np.pad(data['action_chunks'][start:end],((0,pad),(0,0)),mode='edge')
            values=invert(params,states,actions)
            for target,value in zip((z,rebuilt,errors),values):target[start:end]=np.asarray(value)[:end-start]
            if start==0 or end==n or (start//args.batch_size)%20==0:
                print(f'Shifted inverse {end}/{n}',flush=True)
        del params
        norms,new_stats=statistics(z,errors)
        np.savez_compressed(out/'all_latents.npz',**data,z_demo=z,reconstructed_actions=rebuilt,
                            reconstruction_max_abs=errors,latent_norm=norms,inverse_finite=np.isfinite(z).all(1),
                            reverse_query_times=np.asarray(reverse_times))
        finite_idx=np.flatnonzero(np.isfinite(norms))
        maximum=int(finite_idx[np.argmax(norms[finite_idx])]) if len(finite_idx) else None
        summary=dict(checkpoint=str(path),dataset=str(args.dataset.resolve()),checkpoint_sha256=checkpoint_sha,
                     dataset_sha256=dataset_sha,platform=args.platform,batch_size=args.batch_size,
                     num_demos=len(lengths),num_raw_transitions=sum(lengths.values()),num_valid_chunks=available,
                     horizon=horizon,latent_dimension=dimension,flow_steps=config['flow_steps'],dt=1/config['flow_steps'],
                     inverse_method=FLOW_INVERSE_METHOD,reverse_time_shift=1,
                     reverse_query_times=reverse_times,forward_query_times=[i/config['flow_steps'] for i in range(config['flow_steps'])],
                     no_latent_filtering=True,clustering_performed=False,statistics=new_stats,
                     maximum_finite_sample=dict(source_row=maximum,demo_id=int(data['demo_id'][maximum]),
                                                timestep=int(data['timestep'][maximum]),latent_norm=float(norms[maximum]),
                                                roundtrip_max_abs=float(errors[maximum])) if maximum is not None else None)
        baseline=args.output/run.name/f'all_demo_clusters_step_{step}_seed_{args.seed}'/'all_latents.npz'
        old_norms=None
        if baseline.exists():
            identity_path=baseline.parent/'inversion_identity.json'
            identity=json.loads(identity_path.read_text())
            if identity['checkpoint_sha256']!=checkpoint_sha or identity['dataset_sha256']!=dataset_sha:
                raise ValueError('Baseline checkpoint/dataset identity mismatch')
            with np.load(baseline) as old:
                np.testing.assert_array_equal(old['demo_id'],data['demo_id'])
                np.testing.assert_array_equal(old['timestep'],data['timestep'])
                np.testing.assert_array_equal(old['action_chunks'],data['action_chunks'])
                old_norms,old_stats=statistics(old['z_demo'],old['reconstruction_max_abs'])
                old_errors=old['reconstruction_max_abs'].copy()
            old_extreme=old_norms>old_stats['ten_median_threshold']
            paired=[dict(source_row=int(i),demo_id=int(data['demo_id'][i]),timestep=int(data['timestep'][i]),
                         old_norm=float(old_norms[i]),new_norm=float(norms[i]),
                         old_roundtrip_max_abs=float(old_errors[i]),new_roundtrip_max_abs=float(errors[i])) for i in np.flatnonzero(old_extreme)]
            summary.update(baseline_path=str(baseline),baseline_statistics=old_stats,
                           old_extreme_samples=paired,
                           new_count_above_old_extreme_threshold=int((norms>old_stats['ten_median_threshold']).sum()))
            with (out/'old_extreme_samples_comparison.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        order=np.argsort(-np.where(np.isnan(norms),np.inf,norms))
        with (out/'largest_100_norms.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['rank','source_row','demo_id','timestep','latent_norm','old_latent_norm','roundtrip_max_abs'])
            writer.writerows((rank+1,int(i),int(data['demo_id'][i]),int(data['timestep'][i]),float(norms[i]),
                             float(old_norms[i]) if old_norms is not None else '',float(errors[i])) for rank,i in enumerate(order[:100]))
        with (out/'per_demo_statistics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['demo_id','raw_transitions','valid_z','nonfinite_z','median_norm','max_finite_norm','count_norm_gt_10','count_norm_gt_100','count_norm_gt_1000'])
            for demo,length in sorted(lengths.items()):
                values=norms[data['demo_id']==demo];finite=values[np.isfinite(values)]
                writer.writerow([demo,length,len(values),int((~np.isfinite(values)).sum()),float(np.median(finite)),float(finite.max()),int((values>10).sum()),int((values>100).sum()),int((values>1000).sum())])
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        fig,axes=plt.subplots(1,3 if old_norms is not None else 2,figsize=(17 if old_norms is not None else 12,5))
        positive=norms[np.isfinite(norms)&(norms>0)]
        old_positive=old_norms[np.isfinite(old_norms)&(old_norms>0)] if old_norms is not None else positive
        bins=np.geomspace(min(positive.min(),old_positive.min()),max(positive.max(),old_positive.max())*1.01,100)
        if old_norms is not None:axes[0].hist(old_positive,bins=bins,histtype='step',label='Old t=1,...,0.1',linewidth=1.5)
        axes[0].hist(positive,bins=bins,histtype='step',label='New t=0.9,...,0',linewidth=1.5)
        axes[0].set(xscale='log',yscale='log',xlabel='||z|| (log)',ylabel='Count (log)',title='All demo latent norms');axes[0].legend()
        axes[1].scatter(np.arange(n),norms,s=1,alpha=.4)
        axes[1].set(yscale='log',xlabel='Source row (demo/time order)',ylabel='New ||z|| (log)',title=f'New grid: maximum norm={positive.max():.4g}')
        if old_norms is not None:
            both=np.isfinite(old_norms)&np.isfinite(norms)&(old_norms>0)&(norms>0)
            axes[2].scatter(old_norms[both],norms[both],s=1,alpha=.25)
            axes[2].plot([bins[0],bins[-1]],[bins[0],bins[-1]],'k--',linewidth=.8)
            axes[2].set(xscale='log',yscale='log',xlabel='Old ||z||',ylabel='New ||z||',title='Same transitions: old vs new')
        fig.suptitle(f'Square: ALL {len(lengths)} demos, {n} valid chunks; shifted reverse times only')
        fig.tight_layout();fig.savefig(out/'norm_comparison.png',dpi=180);plt.close(fig)
        lines=['# 全部Square demo：反解时间改为0.9到0','',
               f'Checkpoint：`{path}`。数据集：`{args.dataset}`。',
               f'共{len(lengths)}条demo、{sum(lengths.values()):,}个原始transition，全部{n:,}个有效完整动作窗口反解为{dimension}维z。',
               f'反解网络查询时间：{reverse_times}；共{config["flow_steps"]}次更新，每次x←x−0.1v。',
               '固定状态，不裁剪z，不过滤异常值，不重新聚类。末尾不足4步或跨done的窗口按原loader规则排除。',
               '重建始终用原decoder：查询t=0,…,0.9，无中间裁剪、最终动作裁剪至[-1,1]。','',
               '## 模长对比','', '| 指标 | 旧网格1→0.1 | 新网格0.9→0 |','|---|---:|---:|']
        old_stats=summary.get('baseline_statistics',{})
        for q in (.5,.9,.99,.999,1.):
            old_value=old_stats.get('norm_quantiles',{}).get(str(q))
            lines.append(f'| 模长分位数{q:g} | {old_value} | {new_stats["norm_quantiles"][str(q)]} |')
        for threshold in (10,20,50,100,1000,10000,1000000):
            lines.append(f'| 模长>{threshold}的数量 | {old_stats.get("norm_counts_above",{}).get(str(threshold))} | {new_stats["norm_counts_above"][str(threshold)]} |')
        lines += [f'| 非有限z数量 | {old_stats.get("nonfinite_count")} | {new_stats["nonfinite_count"]} |',
                  f'| 重建误差中位数 | {old_stats.get("roundtrip_median")} | {new_stats["roundtrip_median"]} |',
                  f'| 重建误差95分位 | {old_stats.get("roundtrip_p95")} | {new_stats["roundtrip_p95"]} |','',
                  '## 新网格下最大模长样本','', '```json',json.dumps(summary['maximum_finite_sample'],indent=2),'```','',
                  '旧极端样本的新旧模长和重建误差见old_extreme_samples_comparison.csv；新网格最大100个样本见largest_100_norms.csv；每条轨迹覆盖和统计见per_demo_statistics.csv。',
                  '全部未经筛选的z、原状态/动作、demo ID、时间步和重建误差保存在all_latents.npz。','',
                  '复现：','```bash',f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --invert-all-shifted-times --steps {step} --batch-size {args.batch_size} --platform {args.platform}','```','']
        (out/'REPORT.md').write_text('\n'.join(lines))
        print(json.dumps(dict(statistics=new_stats,maximum=summary['maximum_finite_sample'])),flush=True)
        print(f'All-demo shifted inverse results: {out}',flush=True)


def analyze_latent_coordinate_interventions(args, run, config, paths):
    """Fixed-state, single-coordinate interventions in the raw 28D decoder input.

    Store BOTH a_perturbed-a_expert (requested) and a_perturbed-D(s,z0),
    separating reconstruction bias from intervention effects. Actual actions
    retain final clipping; pre-clipping outputs/Jacobians are diagnostics only.
    """
    import csv
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def digest_file(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()

    horizon = config['horizon_length'] if config['action_chunking'] else 1
    data, available = load_demos(args.dataset, horizon, 0, args.seed)
    primary = np.flatnonzero((data['demo_id']==args.perturb_demo_id) & (data['timestep']==args.perturb_timestep))
    if len(primary)!=1:
        raise ValueError('Primary demo/timestep is missing or has no valid full action window')
    rng = np.random.default_rng(args.seed)
    other_demos = np.setdiff1d(np.unique(data['demo_id']), [args.perturb_demo_id])
    if args.perturb_num_states > len(other_demos)+1:
        raise ValueError('Requested more independent demo states than available demos')
    selected = [int(primary[0])]
    for demo in rng.choice(other_demos, args.perturb_num_states-1, replace=False):
        selected.append(int(rng.choice(np.flatnonzero(data['demo_id']==demo))))
    rows=np.asarray(selected)
    states=data['observations'][rows]
    experts=data['action_chunks'][rows]
    dimension=experts.shape[1]
    action_dim=dimension//horizon
    if action_dim!=7:
        raise ValueError('This coordinate experiment labels Square seven-component actions')
    deltas=np.asarray(sorted(set(args.perturb_deltas)),dtype=np.float32)
    names=('x','y','z','rx','ry','rz','g')
    action_labels=[f'{h}:{name}' for h in range(horizon) for name in names]
    latent_labels=[f'z{d}' for d in range(dimension)]
    K=config['flow_steps'];eps=.001
    chosen_delta=int(np.argmin(np.abs(deltas-.25)))
    dataset_sha=digest_file(args.dataset)
    invert,_,decode,decode_raw=make_functions(config,dimension,args,return_decoder=True)

    def decode_batches(params, batch_states, batch_z):
        clipped=[];raw=[]
        for start in range(0,len(batch_z),args.batch_size):
            end=min(start+args.batch_size,len(batch_z));pad=args.batch_size-(end-start)
            ss=np.pad(batch_states[start:end],((0,pad),(0,0)),mode='edge')
            zz=np.pad(batch_z[start:end],((0,pad),(0,0)),mode='edge')
            # One raw decoder pass; the actual action is its final [-1,1] clip.
            r=np.asarray(decode_raw(params,ss,zz))[:end-start]
            raw.append(r);clipped.append(np.clip(r,-1,1))
        return np.concatenate(clipped),np.concatenate(raw)

    @jax.jit
    def local_jacobians(params,state,z):
        def outputs(latent):
            raw=decode_raw(params,state[None],latent[None])[0]
            return jnp.stack((jnp.clip(raw,-1,1),raw))
        return jax.jacfwd(outputs)(z)

    same_component=np.equal.outer(np.arange(dimension)%7,np.arange(dimension)%7)
    same_step=np.equal.outer(np.arange(dimension)//7,np.arange(dimension)//7)
    arm=np.flatnonzero(np.arange(dimension)%7!=6)

    def matrix_metrics(jac):
        e=np.square(jac.astype(np.float64));total=float(e.sum())
        per_col=e.sum(0);active=np.sqrt(per_col)>1e-4
        diag=np.diag(e)
        arm_matrix=e[np.ix_(arm,arm)]
        return dict(total_sensitivity_energy=total,active_latent_dimensions=int(active.sum()),
                    inactive_latent_dimensions=np.flatnonzero(~active).tolist(),
                    exact_diagonal_energy_fraction=float(diag.sum()/max(total,1e-30)),
                    same_action_component_energy_fraction=float(e[same_component].sum()/max(total,1e-30)),
                    same_chunk_step_energy_fraction=float(e[same_step].sum()/max(total,1e-30)),
                    diagonal_argmax_fraction_active=float(np.mean(np.argmax(np.abs(jac),axis=0)[active]==np.arange(dimension)[active])) if active.any() else None,
                    arm_only_diagonal_energy_fraction=float(np.trace(arm_matrix)/max(float(arm_matrix.sum()),1e-30)),
                    arm_only_same_component_energy_fraction=float(arm_matrix[same_component[np.ix_(arm,arm)]].sum()/max(float(arm_matrix.sum()),1e-30)))

    def draw_matrix(ax,matrix,title,row_labels=None,column_labels=None,normalize=False):
        matrix=matrix.copy()
        if normalize:
            matrix=np.abs(matrix)/np.maximum(np.max(np.abs(matrix),axis=0,keepdims=True),1e-12)
            image=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=1,aspect='auto')
        else:
            vmax=max(float(np.max(np.abs(matrix))),1e-8)
            image=ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        rr=row_labels if row_labels is not None else action_labels
        cc=column_labels if column_labels is not None else latent_labels
        ax.set(xticks=np.arange(len(cc)),xticklabels=cc,yticks=np.arange(len(rr)),yticklabels=rr,
               xlabel='Perturbed latent coordinate (column)',ylabel='Action coordinate (row)',title=title)
        ax.tick_params(axis='x',labelrotation=90,labelsize=6);ax.tick_params(axis='y',labelsize=6)
        if matrix.shape==(dimension,dimension):
            for x in np.arange(6.5,dimension,7):
                ax.axvline(x,color='black',linewidth=.7,alpha=.6);ax.axhline(x,color='black',linewidth=.7,alpha=.6)
        return image

    print(f'Coordinate interventions: {len(rows)} fixed states, {dimension} latent coordinates, deltas={deltas.tolist()}',flush=True)
    print('States: '+str([(int(data['demo_id'][i]),int(data['timestep'][i])) for i in rows]),flush=True)
    for path in (paths if args.steps else paths[-1:]):
        step=int(path.stem.split('_')[-1])
        out=args.output/run.name/FLOW_INVERSE_OUTPUT_DIR/f'latent_coordinate_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True,exist_ok=True)
        checkpoint_sha=digest_file(path)
        with path.open('rb') as f:payload=pickle.load(f)
        params=payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow'];del payload
        z0,rebuilt,errors=[np.asarray(v) for v in invert(params,states,experts)]
        baseline,raw_baseline=decode_batches(params,states,z0)
        np.testing.assert_allclose(baseline,rebuilt,rtol=1e-5,atol=2e-6)
        eye=np.eye(dimension,dtype=np.float32)
        perturbed_z=z0[:,None,None,:]+deltas[None,:,None,None]*eye[None,None,:,:]
        all_states=np.broadcast_to(states[:,None,None,:],(len(rows),len(deltas),dimension,states.shape[1])).reshape(-1,states.shape[1])
        perturbed_actions,raw_actions=decode_batches(params,all_states,perturbed_z.reshape(-1,dimension))
        shape=(len(rows),len(deltas),dimension,dimension)
        perturbed_actions=perturbed_actions.reshape(shape);raw_actions=raw_actions.reshape(shape)
        delta_expert=perturbed_actions-experts[:,None,None,:]
        delta_latent=perturbed_actions-baseline[:,None,None,:]
        raw_delta_latent=raw_actions-raw_baseline[:,None,None,:]
        np.testing.assert_allclose(delta_expert,delta_latent+(baseline-experts)[:,None,None,:],rtol=2e-5,atol=2e-7)
        jacobians=np.stack([np.asarray(local_jacobians(params,s,z)) for s,z in zip(states,z0)])
        # Independent small finite-difference check of the autodiff sensitivity.
        fd_z=z0[:,None,None,:]+np.array([-eps,eps],np.float32)[None,:,None,None]*eye[None,None,:,:]
        fd_states=np.broadcast_to(states[:,None,None,:],(len(rows),2,dimension,states.shape[1])).reshape(-1,states.shape[1])
        fd_actions,fd_raw=decode_batches(params,fd_states,fd_z.reshape(-1,dimension))
        fd_actions=fd_actions.reshape(len(rows),2,dimension,dimension)
        fd_raw=fd_raw.reshape(len(rows),2,dimension,dimension)
        fd_jac=np.swapaxes((fd_actions[:,1]-fd_actions[:,0])/(2*eps),1,2)
        fd_raw_jac=np.swapaxes((fd_raw[:,1]-fd_raw[:,0])/(2*eps),1,2)
        records=[]
        for b,row in enumerate(rows):
            record=dict(demo_id=int(data['demo_id'][row]),timestep=int(data['timestep'][row]),source_row=int(row),
                        primary=(b==0),latent_norm=float(np.linalg.norm(z0[b].astype(np.float64))),
                        reconstruction_max_abs=float(np.max(np.abs(baseline[b]-experts[b]))),
                        reconstructed_clipped_coordinates=np.flatnonzero(np.abs(raw_baseline[b])>=1).tolist(),
                        clipped=matrix_metrics(jacobians[b,0]),preclip=matrix_metrics(jacobians[b,1]),
                        finite_difference_epsilon=eps,
                        finite_difference_relative_fro_error=float(np.linalg.norm(fd_jac[b]-jacobians[b,0])/max(np.linalg.norm(jacobians[b,0]),1e-12)),
                        preclip_fd_relative_fro_error=float(np.linalg.norm(fd_raw_jac[b]-jacobians[b,1])/max(np.linalg.norm(jacobians[b,1]),1e-12)))
            records.append(record)
            print(f'demo_{record["demo_id"]} t={record["timestep"]}: diagonal energy={record["clipped"]["exact_diagonal_energy_fraction"]:.3f}, same-component energy={record["clipped"]["same_action_component_energy_fraction"]:.3f}, roundtrip={record["reconstruction_max_abs"]:.4f}',flush=True)
        np.savez_compressed(out/'interventions.npz',demo_id=data['demo_id'][rows],timestep=data['timestep'][rows],source_rows=rows,
                            states=states,expert_actions=experts,z0=z0,baseline_actions=baseline,raw_baseline_actions=raw_baseline,
                            deltas=deltas,perturbed_z=perturbed_z,perturbed_actions=perturbed_actions,raw_perturbed_actions=raw_actions,
                            delta_vs_expert=delta_expert,delta_vs_rebuilt=delta_latent,raw_delta_vs_rebuilt=raw_delta_latent,
                            jacobian_clipped=jacobians[:,0],jacobian_preclip=jacobians[:,1],finite_difference_jacobian=fd_jac,
                            finite_difference_preclip_jacobian=fd_raw_jac,finite_difference_epsilon=eps)
        primary_dir=out/f'demo_{records[0]["demo_id"]}_t_{records[0]["timestep"]}'
        primary_dir.mkdir(exist_ok=True)
        for label,matrix in (('jacobian_clipped',jacobians[0,0]),('jacobian_preclip',jacobians[0,1]),
                             ('requested_delta_expert',delta_expert[0,chosen_delta].T),('delta_rebuilt',delta_latent[0,chosen_delta].T)):
            with (primary_dir/f'{label}.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['action_row/latent_column']+latent_labels)
                writer.writerows([action_labels[i]]+matrix[i].tolist() for i in range(dimension))
        with (primary_dir/'perturbations_long.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['delta','latent_dimension','action_step','action_component','expert_action','rebuilt_action','perturbed_action','delta_vs_expert','delta_vs_rebuilt','preclip_action'])
            for q,delta in enumerate(deltas):
                for i in range(dimension):
                    for j in range(dimension):
                        writer.writerow([float(delta),i,j//7,names[j%7],float(experts[0,j]),float(baseline[0,j]),float(perturbed_actions[0,q,i,j]),float(delta_expert[0,q,i,j]),float(delta_latent[0,q,i,j]),float(raw_actions[0,q,i,j])])
        fig,axes=plt.subplots(2,2,figsize=(16,14))
        matrices=[delta_expert[0,chosen_delta].T,delta_latent[0,chosen_delta].T,jacobians[0,0],jacobians[0,0,:7,:7]]
        titles=[f'Requested: a(z0+delta*e_i) - expert; delta={deltas[chosen_delta]:g}',
                f'Intervention only: a(z0+delta*e_i) - a(z0); delta={deltas[chosen_delta]:g}',
                'Local action sensitivity da/dz (actual clipped decoder)', 'First action vs first seven latent coordinates']
        for index,(ax,matrix,title) in enumerate(zip(axes.flat,matrices,titles)):
            img=draw_matrix(ax,matrix,title,action_labels[:7] if index==3 else None,latent_labels[:7] if index==3 else None)
            fig.colorbar(img,ax=ax,shrink=.8)
        fig.suptitle(f'Fixed Square state demo_{records[0]["demo_id"]}, t={records[0]["timestep"]}; columns=latent, rows=action; rotation components are axis-angle increments')
        fig.tight_layout();fig.savefig(primary_dir/'action_effects.png',dpi=180);fig.savefig(primary_dir/'action_effects.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(20,7))
        for ax,matrix,title,norm in zip(axes,[jacobians[0,0],jacobians[0,1],jacobians[0,0]],
                                      ['Actual action Jacobian','Pre-final-clip Jacobian (diagnostic)','Absolute sensitivity / column maximum'],[False,False,True]):
            img=draw_matrix(ax,matrix,title,normalize=norm);fig.colorbar(img,ax=ax,shrink=.8)
        fig.tight_layout();fig.savefig(primary_dir/'jacobian_diagnostics.png',dpi=180);plt.close(fig)
        # Direct response curves for the first seven input coordinates/first action.
        fig,axes=plt.subplots(2,4,figsize=(17,8))
        curve_delta=np.r_[deltas[deltas<0],0.,deltas[deltas>0]]
        for i,ax in enumerate(axes.flat):
            if i==7:ax.axis('off');continue
            curve=np.insert(delta_latent[0,:,i,:7],int((deltas<0).sum()),0,axis=0)
            for j,name in enumerate(names):ax.plot(curve_delta,curve[:,j],marker='.',label=name,linewidth=1.2)
            ax.axhline(0,color='gray',linewidth=.5);ax.axvline(0,color='gray',linewidth=.5)
            ax.set(title=f'Perturb z[{i}] only',xlabel='delta',ylabel='First-action change vs a(z0)');ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[7].legend(handles,labels,loc='center',ncol=2)
        fig.suptitle('Signed one-coordinate interventions at the SAME offline state; no state changes')
        fig.tight_layout();fig.savefig(primary_dir/'response_curves.png',dpi=180);plt.close(fig)
        plot_rows=(len(rows)+3)//4
        fig,axes=plt.subplots(plot_rows,4,figsize=(18,4.5*plot_rows),squeeze=False)
        vmax=max(float(np.quantile(np.abs(jacobians[:,0]),.99)),1e-6)
        for b,ax in enumerate(axes.flat):
            if b>=len(rows):ax.axis('off');continue
            ax.imshow(jacobians[b,0],cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
            ax.set(title=f'demo_{records[b]["demo_id"]} t={records[b]["timestep"]}; diag={records[b]["clipped"]["exact_diagonal_energy_fraction"]:.2f}',xlabel='Latent column',ylabel='Action row')
            for line in (6.5,13.5,20.5):ax.axvline(line,color='black',linewidth=.4);ax.axhline(line,color='black',linewidth=.4)
        fig.suptitle('Independent fixed-state Jacobians; SAME color scale, K=10 inverse times 0.9,...,0')
        fig.tight_layout();fig.savefig(out/'states_overview.png',dpi=180);plt.close(fig)
        summary=dict(checkpoint=str(path),checkpoint_sha256=checkpoint_sha,dataset=str(args.dataset.resolve()),dataset_sha256=dataset_sha,
                     inverse_method=FLOW_INVERSE_METHOD,reverse_query_times=list(inverse_time_grid(K)),forward_query_times=[i/K for i in range(K)],
                     seed=args.seed,num_states=len(rows),horizon=horizon,action_dim=7,latent_dimension=dimension,deltas=deltas.tolist(),
                     matrix_orientation='rows=flattened output action; columns=perturbed latent dimension; indices zero-based',
                     action_component_labels=names,rotation_convention='OSC_POSE axis-angle increments, not Euler angles',
                     action_units='Normalized controller commands, not measured end-effector motion',
                     definitions=dict(requested='D(s,z0+delta*e_i)-expert_action',intervention_only='D(s,z0+delta*e_i)-D(s,z0)',
                                      jacobian='autodiff derivative of ORIGINAL decoder with final clipping; preclip saved separately'),
                     states=records)
        canonical=args.output/run.name/f'all_demo_inverse_step_{step}_time_shift_1'
        if (canonical/'summary.json').exists():
            cached_meta=json.loads((canonical/'summary.json').read_text())
            if cached_meta['checkpoint_sha256']==checkpoint_sha and cached_meta['dataset_sha256']==dataset_sha:
                with np.load(canonical/'all_latents.npz') as cache:
                    np.testing.assert_array_equal(cache['demo_id'][rows],data['demo_id'][rows])
                    np.testing.assert_array_equal(cache['timestep'][rows],data['timestep'][rows])
                    np.testing.assert_allclose(cache['z_demo'][rows],z0,rtol=2e-5,atol=2e-5)
                    summary['max_inverse_difference_to_validated_full_dataset']=float(np.max(np.abs(cache['z_demo'][rows]-z0)))
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (out/'state_metrics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['demo_id','timestep','reconstruction_max_abs','clipped_diagonal_energy','clipped_same_component_energy','clipped_same_step_energy','preclip_diagonal_energy','preclip_same_component_energy','arm_only_diagonal_energy','active_latent_dimensions','fd_relative_error'])
            for r in records:writer.writerow([r['demo_id'],r['timestep'],r['reconstruction_max_abs'],r['clipped']['exact_diagonal_energy_fraction'],r['clipped']['same_action_component_energy_fraction'],r['clipped']['same_chunk_step_energy_fraction'],r['preclip']['exact_diagonal_energy_fraction'],r['preclip']['same_action_component_energy_fraction'],r['clipped']['arm_only_diagonal_energy_fraction'],r['clipped']['active_latent_dimensions'],r['finite_difference_relative_fro_error']])
        lines=['# 实验1：固定状态，逐维扰动latent，检查动作坐标对应关系','',
               f'Checkpoint：`{path}`；数据集：`{args.dataset}`。',
               f'主样本：demo_{records[0]["demo_id"]}, t={records[0]["timestep"]}，动作窗口包含连续{horizon}步。另取{len(rows)-1}个其他demo的固定状态作重复。',
               f'使用当前共享反解t=0.9,…,0，得到z0；每次只改变28维中的一个坐标，delta={deltas.tolist()}。每次解码均固定对应状态s，保持原10步前向和最终[-1,1]裁剪，不对z施加tanh/裁剪。','',
               '## 如何看图','',
               f'[主样本影响矩阵](demo_{records[0]["demo_id"]}_t_{records[0]["timestep"]}/action_effects.png)：行是动作输出，列是被扰动的z坐标；每7行/列是一组。',
               '左上是用户定义a′−专家a0；右上减去重建动作D(s,z0)，仅显示扰动的影响。两者之差就是反解的重建偏差，不能把该偏差当成latent耦合。',
               '左下为局部Jacobian，右下放大第一步动作与前7个latent坐标；对角能量越集中，越支持相同索引的局部对应。',
               f'[正负扰动响应曲线](demo_{records[0]["demo_id"]}_t_{records[0]["timestep"]}/response_curves.png)：每格只改变一个z，线条为第一步动作的7个分量。',
               f'[裁剪诊断](demo_{records[0]["demo_id"]}_t_{records[0]["timestep"]}/jacobian_diagnostics.png)：对照原decoder与最终裁剪前的敏感度；饱和行可能有零响应，并不表示原始flow不受该latent影响。','',
               '## 指标定义','',
               '精确对角能量=ΣJ[i,i]^2/ΣJ[j,i]^2；同动作分量能量允许跨4步但要求j%7=i%7；同时间步能量要求j//7=i//7。还报告排除夹爪后的24×24矩阵指标，避免夹爪主导。',
               '局部Jacobian另用±0.001的中心差分核对；clip边界或非线性可能导致差分与点导数不同。','',
               '| demo | t | 重建最大误差 | 精确对角能量 | 同动作分量能量 | 同时间步能量 | 去夹爪对角能量 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for r in records:lines.append(f'| {r["demo_id"]} | {r["timestep"]} | {r["reconstruction_max_abs"]:.4f} | {r["clipped"]["exact_diagonal_energy_fraction"]:.3f} | {r["clipped"]["same_action_component_energy_fraction"]:.3f} | {r["clipped"]["same_chunk_step_energy_fraction"]:.3f} | {r["clipped"]["arm_only_diagonal_energy_fraction"]:.3f} |')
        lines += ['', 'x/y/z与rx/ry/rz/g是归一化控制命令；Square OSC_POSE的旋转分量是轴角增量，不是欧拉角，也不是实际测量到的末端位姿变化。',
                  '该实验检验固定状态下latent与动作坐标的局部控制对应，不能单独证明广义空间感知或任务理解。','',
                  '`interventions.npz`保存全部状态、专家动作、z0、每次扰动的z/动作、两种Δa、raw诊断和Jacobian；主样本目录另含逐项CSV。',
                  '复现：','```bash',f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --latent-coordinate-experiment --steps {step} --perturb-demo-id {args.perturb_demo_id} --perturb-timestep {args.perturb_timestep} --perturb-num-states {args.perturb_num_states} --batch-size {args.batch_size} --platform {args.platform}','```','']
        (out/'REPORT.md').write_text('\n'.join(lines))
        print(f'Latent coordinate intervention results: {out}',flush=True)


def analyze_jacobian_minus_identity(source_dir):
    """Postprocess saved Jacobians only; never invert or decode new samples.

    The pre-final-clip J-I diagnoses the integrated flow residual derivative.
    The actual clipped decoder's J-I also contains clipping effects. Removing
    the diagonal/time-diagonal separates coupling from the cancelling -I term.
    """
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source_dir=Path(source_dir).resolve()
    source=json.loads((source_dir/'summary.json').read_text())
    input_file=source_dir/'interventions.npz'
    with np.load(input_file) as data:
        clipped=data['jacobian_clipped'].astype(np.float64)
        preclip=data['jacobian_preclip'].astype(np.float64)
        demo_ids=data['demo_id'];timesteps=data['timestep']
    if clipped.shape!=preclip.shape or clipped.ndim!=3 or clipped.shape[1]!=clipped.shape[2]:
        raise ValueError('Expected matching [states, action_dim, latent_dim] square Jacobians')
    if not np.isfinite(clipped).all() or not np.isfinite(preclip).all():
        raise ValueError('Nonfinite source Jacobian')
    B,D,_=clipped.shape;horizon=source['horizon'];C=source['action_dim']
    if D!=horizon*C or C!=7:
        raise ValueError('Expected four-step Square coordinate layout')
    out=source_dir/'jacobian_minus_identity';out.mkdir(exist_ok=True)
    identity=np.eye(D,dtype=np.float64);identity_norm=float(np.sqrt(D))
    diagonal=np.eye(D,dtype=bool)
    same_component=np.equal.outer(np.arange(D)%C,np.arange(D)%C)
    same_time=np.equal.outer(np.arange(D)//C,np.arange(D)//C)
    off_time=~same_time;off_diagonal=~diagonal
    names=('x','y','z','rx','ry','rz','g')
    labels=[f'{h}:{name}' for h in range(horizon) for name in names]
    records=[];saved=dict(identity=identity,off_time_mask=off_time,off_diagonal_mask=off_diagonal,
                          same_component_mask=same_component,demo_id=demo_ids,timestep=timesteps)

    def energy_fraction(matrix,mask):
        energy=matrix*matrix
        return float(energy[mask].sum()/max(float(energy.sum()),1e-30))

    for variant,jacobians in (('clipped',clipped),('preclip',preclip)):
        residual=jacobians-identity[None]
        offdiag=residual*off_diagonal[None]
        cross=residual*off_time[None]
        saved.update({f'{variant}_J':jacobians,f'{variant}_J_minus_I':residual,
                      f'{variant}_off_diagonal':offdiag,f'{variant}_cross_time':cross})
        # Off-diagonal entries cannot be created by subtracting I.
        np.testing.assert_array_equal(residual[:,off_diagonal],jacobians[:,off_diagonal])
        np.testing.assert_array_equal(cross[:,off_time],jacobians[:,off_time])
        np.testing.assert_allclose(residual+identity[None],jacobians,rtol=1e-12,atol=2e-16)
        for b in range(B):
            J=jacobians[b];L=residual[b];O=offdiag[b];T=cross[b]
            j_norm=float(np.linalg.norm(J));l_norm=float(np.linalg.norm(L));cross_norm=float(np.linalg.norm(T))
            # Frobenius expansion explicitly exposes cancellation with I.
            cross_term=float(2*np.sum(identity*L))
            np.testing.assert_allclose(j_norm*j_norm,D+l_norm*l_norm+cross_term,atol=2e-13,rtol=1e-12)
            record=dict(variant=variant,demo_id=int(demo_ids[b]),timestep=int(timesteps[b]),primary=(b==0),
                        J_frobenius=j_norm,J_minus_I_frobenius=l_norm,identity_frobenius=identity_norm,
                        J_norm_over_I=j_norm/identity_norm,residual_norm_over_I=l_norm/identity_norm,
                        mean_J_diagonal=float(np.trace(J)/D),
                        residual_cosine_with_I=float(np.sum(identity*L)/max(identity_norm*l_norm,1e-30)),
                        J_same_component_energy_fraction=energy_fraction(J,same_component),
                        residual_same_component_energy_fraction=energy_fraction(L,same_component),
                        residual_diagonal_energy_fraction=energy_fraction(L,diagonal),
                        residual_cross_time_energy_fraction=energy_fraction(L,off_time),
                        off_diagonal_same_component_energy_fraction=energy_fraction(O,same_component),
                        off_diagonal_cross_time_energy_fraction=energy_fraction(O,off_time),
                        cross_time_frobenius=cross_norm,cross_time_norm_over_I=cross_norm/identity_norm,
                        cross_time_same_component_energy_fraction=energy_fraction(T,same_component),
                        cross_time_fraction_of_J_energy=energy_fraction(J,off_time),
                        identity_cross_term=cross_term)
            records.append(record)
    np.savez_compressed(out/'matrices.npz',**saved)
    with (out/'state_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (out/'component_temporal_blocks.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['variant','demo_id','timestep','component','output_time','input_time','J','J_minus_I','cross_time_entry'])
        for variant in ('clipped','preclip'):
            for b in range(B):
                for c,name in enumerate(names):
                    for h_out in range(horizon):
                        for h_in in range(horizon):
                            r=h_out*C+c;k=h_in*C+c
                            writer.writerow([variant,int(demo_ids[b]),int(timesteps[b]),name,h_out,h_in,
                                             float(saved[f'{variant}_J'][b,r,k]),float(saved[f'{variant}_J_minus_I'][b,r,k]),h_out!=h_in])
    # Display J and its residual on their own signed scales; -I otherwise hides coupling.
    fig,axes=plt.subplots(2,3,figsize=(19,13))
    for row,variant in enumerate(('clipped','preclip')):
        for col,(suffix,title) in enumerate((('J','J'),('J_minus_I','J - I'),('off_diagonal','J - I with coordinate diagonal masked'))):
            matrix=saved[f'{variant}_{suffix}'][0]
            limit=max(float(np.max(np.abs(matrix))),1e-10)
            im=axes[row,col].imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            axes[row,col].set(title=f'{variant}: {title}',xlabel='Latent column',ylabel='Action row',
                              xticks=np.arange(0,D,C),xticklabels=np.arange(0,D,C),
                              yticks=np.arange(0,D,C),yticklabels=[labels[i] for i in range(0,D,C)])
            for line in np.arange(C-.5,D,C):axes[row,col].axvline(line,color='black',linewidth=.5);axes[row,col].axhline(line,color='black',linewidth=.5)
            fig.colorbar(im,ax=axes[row,col],shrink=.8)
    fig.suptitle(f'Primary demo_{demo_ids[0]}, t={timesteps[0]}; each panel has its OWN color scale')
    fig.tight_layout();fig.savefig(out/'J_minus_I_primary.png',dpi=180);fig.savefig(out/'J_minus_I_primary.pdf');plt.close(fig)
    for variant in ('preclip','clipped'):
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        primary=saved[f'{variant}_J_minus_I'][0]
        common=max(float(np.max(np.abs(primary*off_time))),1e-10)
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
        for c,ax in enumerate(axes.flat):
            if c==7:
                ax.axis('off');ax.text(.05,.85,'Gray diagonal is excluded.\nOff-time entries satisfy (J-I)[h,k] = J[h,k].\nRows: output action time.\nColumns: latent time.\nAll panels share the SAME color scale.\nWeak blocks should be read using annotations.',va='top',fontsize=11);continue
            index=np.arange(horizon)*C+c
            block=primary[np.ix_(index,index)]
            shown=np.ma.array(block,mask=np.eye(horizon,dtype=bool))
            im=ax.imshow(shown,cmap=cmap,vmin=-common,vmax=common)
            ax.set(title=f'{names[c]} -> {names[c]}',xticks=range(horizon),yticks=range(horizon),xlabel='Input latent time',ylabel='Output action time')
            for h_out in range(horizon):
                for h_in in range(horizon):
                    if h_out==h_in:ax.text(h_in,h_out,'--',ha='center',va='center',color='gray')
                    else:ax.text(h_in,h_out,f'{block[h_out,h_in]:+.2g}',ha='center',va='center',fontsize=8)
        fig.suptitle(f'Primary {variant} flow: cross-time 4x4 blocks of J-I; identity has ZERO entries here')
        fig.tight_layout();fig.savefig(out/f'cross_time_blocks_{variant}.png',dpi=180);fig.savefig(out/f'cross_time_blocks_{variant}.pdf');plt.close(fig)
    pre=[r for r in records if r['variant']=='preclip']
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    xx=np.arange(B);state_labels=[f'{int(d)}:{int(t)}' for d,t in zip(demo_ids,timesteps)]
    axes[0].bar(xx-.18,[r['J_norm_over_I'] for r in pre],width=.36,label='||J|| / ||I||')
    axes[0].bar(xx+.18,[r['residual_norm_over_I'] for r in pre],width=.36,label='||J-I|| / ||I||')
    axes[0].set(title='Preclip norms: cancellation is possible',xticks=xx,xticklabels=state_labels,ylabel='Frobenius norm ratio');axes[0].tick_params(axis='x',rotation=45);axes[0].legend(fontsize=8)
    for offset,key,label in ((-.24,'J_same_component_energy_fraction','J'),(0,'residual_same_component_energy_fraction','J-I'),(.24,'off_diagonal_same_component_energy_fraction','offdiag(J-I)')):
        axes[1].bar(xx+offset,[r[key] for r in pre],width=.24,label=label)
    axes[1].set(title='Same-component energy: inspect the diagonal bias',xticks=xx,xticklabels=state_labels,ylim=(0,1.05));axes[1].tick_params(axis='x',rotation=45);axes[1].legend(fontsize=8)
    cross=saved['preclip_cross_time']
    component_energy=np.mean((cross.reshape(B,horizon,C,horizon,C)**2).sum(axis=(1,3)),axis=0)
    component_fraction=component_energy/np.maximum(component_energy.sum(0,keepdims=True),1e-30)
    im=axes[2].imshow(component_fraction,cmap='Blues',vmin=0,vmax=1)
    axes[2].set(title='ONLY cross-time energy; pooled over states',xticks=range(C),xticklabels=names,yticks=range(C),yticklabels=names,xlabel='Latent component group',ylabel='Action component')
    for r in range(C):
        for c in range(C):axes[2].text(c,r,f'{component_fraction[r,c]:.0%}',ha='center',va='center',fontsize=8,color='white' if component_fraction[r,c]>.6 else 'black')
    fig.colorbar(im,ax=axes[2],shrink=.8)
    fig.tight_layout();fig.savefig(out/'structure_summary.png',dpi=180);plt.close(fig)
    summary=dict(source_directory=str(source_dir),source_checkpoint=source['checkpoint'],source_dataset=source['dataset'],
                 inverse_method=source['inverse_method'],reverse_query_times=source['reverse_query_times'],
                 num_states=B,dimension=D,identity_frobenius=identity_norm,primary_variant_for_flow_interpretation='preclip',
                 definitions=dict(residual='L=J-I',norm_ratio='||J-I||_F/||I||_F, not an additive contribution percentage',
                                  component_correspondence='sum of squared entries with output_index%7=input_index%7 / total squared entries',
                                  off_diagonal='coordinate diagonal removed from J-I',cross_time='only entries with output_index//7 != input_index//7',
                                  clipping='clipped J-I includes clipping; preclip J-I is the integrated flow residual derivative'),
                 states=records,pooled_preclip_cross_time_component_energy=component_energy.tolist(),
                 pooled_preclip_cross_time_component_fraction=component_fraction.tolist())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    primary=pre[0]
    lines=['# 补充：J-I与跨时间耦合诊断','',
           '**仅对已有8个状态的Jacobian做矩阵后处理；没有重复反解、扰动、decoder推理或训练。**','',
           '定义L=J−I，I为28×28恒等矩阵，||I||F=√28。以裁剪前J为Flow学习残差的主要诊断；实际动作的裁剪版并列保存。','',
           '## 主样本demo_226,t=70','',
           '| 指标 | 裁剪前Flow | 实际裁剪动作 |','|---|---:|---:|']
    clipped_primary=records[0]
    fields=[('||J||F / ||I||F','J_norm_over_I'),('||J-I||F / ||I||F','residual_norm_over_I'),
            ('J同分量能量比例','J_same_component_energy_fraction'),('J-I同分量能量比例','residual_same_component_energy_fraction'),
            ('J-I精确对角能量比例','residual_diagonal_energy_fraction'),('去对角后的同分量能量比例','off_diagonal_same_component_energy_fraction'),
            ('跨时间项的同分量能量比例','cross_time_same_component_energy_fraction'),('跨时间项能量占J总能量','cross_time_fraction_of_J_energy'),
            ('跨时间项范数 / ||I||F','cross_time_norm_over_I'),('J-I与I的余弦相似度','residual_cosine_with_I')]
    for label,key in fields:lines.append(f'| {label} | {primary[key]:.6f} | {clipped_primary[key]:.6f} |')
    lines += ['', '## 如何解释','',
              '||J-I||/||I||不是“学习贡献占百分之几”。J=I+(J-I)中两部分可以强烈抵消；这里应同时读||J||、残差与I的方向及交叉项。',
              'J-I的大负对角表示局部抵消恒等通路。减掉I本身也会增强同分量对角，因此J-I总体分量对应率很高，不能单独作为学习到语义结构的证据。',
              '更关键的是去除对角后的部分，尤其跨动作时间步项：恒等矩阵这些位置均为0，所以其数值与J中的对应项逐元素相同，不能由减I凭空产生。',
              '实际裁剪版中，饱和输出的J行可以为0，减I会在相应对角产生-1。这部分不能全部归于速度网络学习；裁剪前版本避免该混淆。',
              '这些指标描述局部映射的非恒等导数及耦合，不能证明环境空间语义理解；也不是把每个Euler步的贡献单独分解。','',
              '## 各状态裁剪前结果','',
              '| demo | t | ||J-I||/||I|| | J-I同分量率 | 去对角同分量率 | 跨时间同分量率 | 跨时间能量/J能量 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in pre:lines.append(f'| {r["demo_id"]} | {r["timestep"]} | {r["residual_norm_over_I"]:.4f} | {r["residual_same_component_energy_fraction"]:.1%} | {r["off_diagonal_same_component_energy_fraction"]:.1%} | {r["cross_time_same_component_energy_fraction"]:.1%} | {r["cross_time_fraction_of_J_energy"]:.1%} |')
    lines += ['', '## 图与数据','',
              '[J、J-I、去对角残差对照](J_minus_I_primary.png)；[裁剪前4×4跨时间块](cross_time_blocks_preclip.png)；[实际动作4×4块](cross_time_blocks_clipped.png)；[8状态汇总](structure_summary.png)。',
              '4×4图中行表示输出动作时间，列表示latent的时间位置；灰色对角被掩去，只看跨时间项；七个分量共用同一色标，弱分量请结合标注数值。',
              '输入组名x/y/z/rx/ry/rz/g只是按索引%7对齐的标签；旋转是轴角增量。',
              '`matrices.npz`保存原J、I、J-I、去对角/跨时间矩阵与掩码；`state_metrics.csv`保存16组状态指标；`component_temporal_blocks.csv`逐个保存全部4×4块数值。','',
              '复现：','```bash',f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --jacobian-minus-identity-from {source_dir}','```','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('Primary preclip metrics: '+json.dumps(primary),flush=True)
    print(f'J-I postprocessing results: {out}',flush=True)
    return out


def analyze_latent_action_distances(args, run, config, paths):
    """Experiment 3: within-state distances around freshly inverted demo chunks.

    Uniformly select valid state rows without filtering reconstruction errors.
    Shared random sphere directions isolate radial and cross-state effects.
    All action distances reference D(s,z0), never the expert action. Designed
    radius ties are preserved for the primary Spearman; measured-dz Spearman
    is also saved to expose floating-point tie breaking.
    """
    import csv
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    def file_digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()

    def corr(x,y):
        if np.ptp(y)==0:return None
        value=float(spearmanr(np.asarray(x).reshape(-1),np.asarray(y).reshape(-1)).statistic)
        return value if np.isfinite(value) else None

    def distribution(values,prefix):
        values=np.asarray(values,dtype=np.float64).reshape(-1)
        d={prefix+'_mean':float(values.mean()),prefix+'_std':float(values.std(ddof=1)) if len(values)>1 else 0.,
           prefix+'_min':float(values.min()),prefix+'_max':float(values.max())}
        for q in (.05,.25,.5,.75,.95):d[f'{prefix}_q{int(q*100):02d}']=float(np.quantile(values,q))
        d[prefix+'_cv']=d[prefix+'_std']/d[prefix+'_mean'] if d[prefix+'_mean']>0 else None
        d[prefix+'_q95_over_q05']=d[prefix+'_q95']/d[prefix+'_q05'] if d[prefix+'_q05']>0 else None
        d[prefix+'_max_over_min']=d[prefix+'_max']/d[prefix+'_min'] if d[prefix+'_min']>0 else None
        return d

    def write_csv(path,records):
        with Path(path).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    horizon=config['horizon_length'] if config['action_chunking'] else 1
    data,total=load_demos(args.dataset,horizon,0,args.seed)
    B=args.distance_num_states;M=args.distance_num_directions
    if B>total:raise ValueError('Requested more states than available full windows')
    rng=np.random.default_rng(args.seed)
    rows=np.sort(rng.choice(total,B,replace=False))
    states=data['observations'][rows];expert=data['action_chunks'][rows];D=expert.shape[1]
    radii=np.asarray(sorted(set(args.distance_radii)),dtype=np.float32);R=len(radii)
    directions=rng.normal(size=(M,D));directions=(directions/np.linalg.norm(directions,axis=1,keepdims=True)).astype(np.float32)
    nominal=np.broadcast_to(radii[None,:,None],(B,R,M))
    dataset_sha=file_digest(args.dataset)
    with h5py.File(args.dataset,'r') as f:num_demos=len(f['data'])
    invert,_,decode,decode_raw=make_functions(config,D,args,return_decoder=True)
    print(f'Distance experiment: {B} random states from {num_demos} demos; {M} directions x {R} radii = {M*R} points/state; radii={radii.tolist()}',flush=True)
    for path in (paths if args.steps else paths[-1:]):
        step=int(path.stem.split('_')[-1])
        out=args.output/run.name/FLOW_INVERSE_OUTPUT_DIR/f'latent_action_distance_step_{step}_seed_{args.seed}'
        out.mkdir(parents=True,exist_ok=True)
        checkpoint_sha=file_digest(path)
        with path.open('rb') as f:payload=pickle.load(f)
        params=payload['agent']['bc_flow_network']['params']['modules_actor_bc_flow'];del payload
        z0,base,error=[np.asarray(v) for v in invert(params,states,expert)]
        raw_base=np.asarray(decode_raw(params,states,z0))
        np.testing.assert_allclose(base,np.clip(raw_base,-1,1),atol=3e-6,rtol=2e-5)
        perturbed_z=z0[:,None,None,:]+radii[None,:,None,None]*directions[None,None,:,:]
        batch_states=np.broadcast_to(states[:,None,None,:],(B,R,M,states.shape[1])).reshape(-1,states.shape[1])
        flat_z=perturbed_z.reshape(-1,D);pieces=[]
        for start in range(0,len(flat_z),args.batch_size):
            end=min(start+args.batch_size,len(flat_z));pad=args.batch_size-(end-start)
            ss=np.pad(batch_states[start:end],((0,pad),(0,0)),mode='edge')
            zz=np.pad(flat_z[start:end],((0,pad),(0,0)),mode='edge')
            pieces.append(np.asarray(decode_raw(params,ss,zz))[:end-start])
            if start==0 or end==len(flat_z) or (start//args.batch_size)%10==0:print(f'Decoded {end}/{len(flat_z)} perturbations',flush=True)
        raw_actions=np.concatenate(pieces).reshape(B,R,M,D)
        if not np.isfinite(raw_actions).all():raise ValueError('Nonfinite preclip decoder outputs')
        actions=np.clip(raw_actions,-1,1)
        dz=np.linalg.norm(perturbed_z.astype(np.float64)-z0[:,None,None,:].astype(np.float64),axis=-1)
        da=np.linalg.norm(actions.astype(np.float64)-base[:,None,None,:].astype(np.float64),axis=-1)
        da_raw=np.linalg.norm(raw_actions.astype(np.float64)-raw_base[:,None,None,:].astype(np.float64),axis=-1)
        gain=da/dz;gain_raw=da_raw/dz
        clipped_fraction=np.mean(np.abs(raw_actions)>=1,axis=-1)
        dz_error=float(np.max(np.abs(dz-nominal)))
        np.testing.assert_allclose(dz,nominal,rtol=5e-6,atol=5e-7)
        np.savez_compressed(out/'distance_samples.npz',source_rows=rows,demo_id=data['demo_id'][rows],timestep=data['timestep'][rows],
                            states=states,expert_actions=expert,z0=z0,baseline_actions=base,raw_baseline_actions=raw_base,
                            radii=radii,directions=directions,perturbed_z=perturbed_z,perturbed_actions=actions,raw_perturbed_actions=raw_actions,
                            latent_distances=dz,action_distances=da,preclip_action_distances=da_raw,gain=gain,preclip_gain=gain_raw,
                            clipped_coordinate_fraction=clipped_fraction,reconstruction_max_abs=error)
        radius_records=[];state_radius_records=[];state_records=[]
        reference_radius=int(np.argmin(np.abs(radii-1.)))
        tol=1e-6
        monotone=np.diff(da,axis=1)>=-tol
        monotone_raw=np.diff(da_raw,axis=1)>=-tol
        for r,radius in enumerate(radii):
            rec=dict(radius=float(radius),num_states=B,num_directions_per_state=M,n=B*M,
                     median_radius_over_z0_norm=float(np.median(radius/np.linalg.norm(z0.astype(np.float64),axis=1))),
                     perturbation_rms_per_coordinate=float(radius/np.sqrt(D)),
                     mean_clipped_coordinate_fraction=float(clipped_fraction[:,r].mean()),zero_action_response_count=int((da[:,r]<=1e-8).sum()))
            rec.update(distribution(da[:,r],'action_distance'));rec.update(distribution(gain[:,r],'gain'))
            rec.update(distribution(da_raw[:,r],'preclip_action_distance'));rec.update(distribution(gain_raw[:,r],'preclip_gain'))
            per_state_medians=np.median(gain[:,r],axis=1)
            rec.update(distribution(per_state_medians,'state_median_gain'))
            radius_records.append(rec)
        for b,row in enumerate(rows):
            rec=dict(state_index=b,demo_id=int(data['demo_id'][row]),timestep=int(data['timestep'][row]),source_row=int(row),
                     z0_norm=float(np.linalg.norm(z0[b].astype(np.float64))),reconstruction_max_abs=float(error[b]),
                     baseline_clipped_fraction=float(np.mean(np.abs(raw_base[b])>=1)),
                     spearman_radius_ties=corr(nominal[b],da[b]),spearman_measured_dz=corr(dz[b],da[b]),
                     spearman_preclip=corr(nominal[b],da_raw[b]),
                     adjacent_radius_nondecreasing_fraction=float(monotone[b].mean()),
                     all_radii_nondecreasing_direction_fraction=float(monotone[b].all(axis=0).mean()),
                     adjacent_radius_preclip_nondecreasing_fraction=float(monotone_raw[b].mean()),
                     reference_radius=float(radii[reference_radius]))
            rec.update(distribution(gain[b,reference_radius],'reference_radius_gain'))
            state_records.append(rec)
            for r,radius in enumerate(radii):
                rr=dict(state_index=b,demo_id=rec['demo_id'],timestep=rec['timestep'],radius=float(radius),n_directions=M,
                        mean_clipped_coordinate_fraction=float(clipped_fraction[b,r].mean()),zero_action_response_count=int((da[b,r]<=1e-8).sum()))
                rr.update(distribution(da[b,r],'action_distance'));rr.update(distribution(gain[b,r],'gain'))
                rr.update(distribution(da_raw[b,r],'preclip_action_distance'));rr.update(distribution(gain_raw[b,r],'preclip_gain'))
                state_radius_records.append(rr)
        write_csv(out/'radius_statistics.csv',radius_records)
        write_csv(out/'state_statistics.csv',state_records)
        write_csv(out/'state_radius_statistics.csv',state_radius_records)
        with (out/'samples_long.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['state_index','demo_id','timestep','direction_id','radius','measured_latent_distance','action_distance','gain','preclip_action_distance','preclip_gain','clipped_coordinate_fraction'])
            for b in range(B):
                for r in range(R):
                    for d in range(M):writer.writerow([b,state_records[b]['demo_id'],state_records[b]['timestep'],d,float(radii[r]),float(dz[b,r,d]),float(da[b,r,d]),float(gain[b,r,d]),float(da_raw[b,r,d]),float(gain_raw[b,r,d]),float(clipped_fraction[b,r,d])])
        valid_rhos=np.asarray([r['spearman_radius_ties'] for r in state_records if r['spearman_radius_ties'] is not None])
        if not len(valid_rhos):raise ValueError('All states have constant action-distance response')
        summary=dict(checkpoint=str(path),checkpoint_sha256=checkpoint_sha,dataset=str(args.dataset.resolve()),dataset_sha256=dataset_sha,
                     seed=args.seed,inverse_method=FLOW_INVERSE_METHOD,reverse_query_times=list(inverse_time_grid(config['flow_steps'])),
                     num_dataset_demos=num_demos,num_states=B,num_distinct_sampled_demos=int(len(np.unique(data['demo_id'][rows]))),
                     num_unique_random_directions=M,num_radii=R,num_perturbations_per_state=R*M,total_perturbations=B*R*M,
                     direction_design='Shared Gaussian-normalized sphere directions, reused across all radii AND states; no direction resampling',
                     state_sampling='Uniform without replacement over ALL valid full windows; no reconstruction-error filtering',
                     action_distance_reference='D(s,z0), not expert action',latent_dimension=D,horizon=horizon,radii=radii.tolist(),
                     spearman_primary='Designed radius with exact ties; measured dz correlation also stored to expose float32 tie breaking',
                     max_measured_radius_error=dz_error,pooled_spearman_radius_ties=corr(nominal,da),pooled_spearman_measured_dz=corr(dz,da),
                     pooled_preclip_spearman=corr(nominal,da_raw),valid_spearman_state_count=len(valid_rhos),
                     per_state_spearman_summary=distribution(valid_rhos,'rho'),
                     fraction_states_rho_ge_09=float(np.mean(valid_rhos>=.9)),fraction_states_rho_ge_08=float(np.mean(valid_rhos>=.8)),
                     adjacent_radius_nondecreasing_fraction=float(monotone.mean()),
                     all_radii_nondecreasing_direction_fraction=float(monotone.all(axis=1).mean()),
                     z0_norm_summary=distribution(np.linalg.norm(z0.astype(np.float64),axis=1),'norm'),
                     perturbed_z_norm_summary=distribution(np.linalg.norm(perturbed_z.astype(np.float64),axis=-1),'norm'),
                     reconstruction_summary=distribution(error,'error'),radius_statistics=radius_records,states=state_records)
        canonical=args.output/run.name/f'all_demo_inverse_step_{step}_time_shift_1'
        if (canonical/'summary.json').exists():
            cm=json.loads((canonical/'summary.json').read_text())
            if cm['checkpoint_sha256']==checkpoint_sha and cm['dataset_sha256']==dataset_sha:
                with np.load(canonical/'all_latents.npz') as cache:
                    np.testing.assert_array_equal(cache['demo_id'][rows],data['demo_id'][rows])
                    np.testing.assert_array_equal(cache['timestep'][rows],data['timestep'][rows])
                    np.testing.assert_allclose(z0,cache['z_demo'][rows],rtol=2e-5,atol=3e-5)
                    summary['max_inverse_difference_to_validated_all_demo_cache']=float(np.max(np.abs(z0-cache['z_demo'][rows])))
                    summary['full_dataset_z0_norm_median']=float(np.median(np.linalg.norm(cache['z_demo'].astype(np.float64),axis=1)))
                    summary['full_dataset_z0_norm_max']=float(np.linalg.norm(cache['z_demo'].astype(np.float64),axis=1).max())
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        # Six panels retain state-wise and direction-wise distributions rather than only pooled rho.
        fig,axes=plt.subplots(2,3,figsize=(18,10))
        colors=plt.get_cmap('viridis')(np.linspace(.1,.9,R))
        for r,radius in enumerate(radii):
            axes[0,0].scatter(dz[:,r].reshape(-1),da[:,r].reshape(-1),s=4,alpha=.2,color=colors[r],label=f'r={radius:g}')
        axes[0,0].set(xlabel='Measured ||z-z0||',ylabel='||D(s,z)-D(s,z0)||',title=f'All {B*R*M:,} interventions; pooled rho={summary["pooled_spearman_radius_ties"]:.3f}')
        axes[0,0].legend(fontsize=8)
        for ax,values,title,ylabel in ((axes[0,1],da,'Action-distance distributions','Action distance'),(axes[0,2],gain,'Directional gain distributions','Action distance / latent distance')):
            parts=ax.violinplot([values[:,r].reshape(-1) for r in range(R)],positions=np.arange(R),showmedians=True,showextrema=False)
            for body,color in zip(parts['bodies'],colors):body.set_facecolor(color);body.set_alpha(.65)
            for r in range(R):
                q=np.quantile(values[:,r],[.05,.5,.95]);ax.plot([r,r],[q[0],q[2]],color='black',linewidth=1);ax.scatter(r,q[1],s=15,color='black')
            ax.set(xticks=np.arange(R),xticklabels=[f'{v:g}' for v in radii],xlabel='Radius',ylabel=ylabel,title=title)
        axes[1,0].hist(valid_rhos,bins=20,color='#2675b6',alpha=.8)
        axes[1,0].axvline(np.median(valid_rhos),color='orange',label=f'median={np.median(valid_rhos):.3f}')
        axes[1,0].set(xlabel='Per-state Spearman (radius ties preserved)',ylabel='States',title='Do distance ranks transfer across states?');axes[1,0].legend(fontsize=8)
        med=np.median(gain[:,reference_radius],axis=1);low,high=np.quantile(gain[:,reference_radius],[.1,.9],axis=1)
        order=np.argsort(med)
        axes[1,1].errorbar(np.arange(B),med[order],yerr=[med[order]-low[order],high[order]-med[order]],fmt='.',markersize=3,elinewidth=.5,alpha=.8)
        axes[1,1].set(xlabel='States sorted by median gain',ylabel='Gain (median and q10/q90 across directions)',title=f'SAME directions, radius={radii[reference_radius]:g}')
        for b in range(B):axes[1,2].plot(radii,np.median(da[b],axis=1),color='gray',alpha=.15,linewidth=.6)
        axes[1,2].plot(radii,np.median(da,axis=(0,2)),color='orange',marker='o',label='Pooled median')
        axes[1,2].set(xlabel='Radius',ylabel='Median action distance',title='One curve per fixed state');axes[1,2].legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(f'Square fixed-state latent/action distances | {B} states x {M} directions x {R} radii | original clipped decoder')
        fig.tight_layout();fig.savefig(out/'distance_overview.png',dpi=180);fig.savefig(out/'distance_overview.pdf');plt.close(fig)
        # Automatically choose worst/median/best rank relation and strongest sampled anisotropy.
        rho_order=sorted(range(B),key=lambda b:state_records[b]['spearman_radius_ties'] if state_records[b]['spearman_radius_ties'] is not None else -2)
        spread=[r['reference_radius_gain_q95_over_q05'] if r['reference_radius_gain_q95_over_q05'] is not None else np.inf for r in state_records]
        examples=[rho_order[0],rho_order[len(rho_order)//2],rho_order[-1],int(np.argmax(spread))]
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        for ax,b,title in zip(axes.flat,examples,('Lowest state rho','Median state rho','Highest state rho','Largest sampled directional spread')):
            for r,radius in enumerate(radii):ax.scatter(dz[b,r],da[b,r],s=20,alpha=.7,color=colors[r],label=f'r={radius:g}')
            rec=state_records[b]
            ax.set(title=f'{title}: demo_{rec["demo_id"]}, t={rec["timestep"]}\nrho={rec["spearman_radius_ties"]}, gain q95/q05 at r={radii[reference_radius]:g}: {rec["reference_radius_gain_q95_over_q05"]:.2f}',xlabel='Latent distance',ylabel='Action distance')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'state_examples.png',dpi=180);plt.close(fig)
        # Paired rays make radial monotonicity observable independently of direction changes.
        b=examples[3];fig,axes=plt.subplots(1,2,figsize=(13,5))
        for d in range(M):
            axes[0].plot(radii,da[b,:,d],'-o',markersize=3,alpha=.6)
            axes[1].plot(radii,gain[b,:,d],'-o',markersize=3,alpha=.6)
        axes[0].set(xlabel='Radius',ylabel='Action distance',title='Same random direction along each curve')
        axes[1].set(xlabel='Radius',ylabel='da/dz',title='Gain can differ across directions and radii')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f'Automatically selected anisotropic state: demo_{state_records[b]["demo_id"]}, t={state_records[b]["timestep"]}')
        fig.tight_layout();fig.savefig(out/'direction_rays.png',dpi=180);plt.close(fig)
        lines=['# 实验三：固定状态的Latent–Action距离关系','',
               f'Checkpoint：`{path}`；Square数据集：`{args.dataset}`。',
               f'从全部有效完整动作窗口中均匀无放回抽取{B}个状态，涉及{summary["num_distinct_sampled_demos"]}条demo；不按反解误差筛选。',
               f'每个状态先用最新反解t=0.9,…,0得到z0；{M}个随机单位方向×{R}个半径，共{M*R}个扰动点/状态、总{B*M*R:,}个点。',
               '同一组随机方向在所有半径和状态间复用，以分离半径和状态的影响；不是100个独立方向。冻结Flow，无z裁剪/tanh/优化；动作保持原decoder最终[-1,1]裁剪。',
               '**动作距离始终参考a_base=D(s,z0)，绝不参考专家动作。** 专家动作仅用于反解及重建诊断。',
               '主Spearman使用设计半径的并列秩；实际dz只因float32舍入偏离设计半径，最大偏差见summary。也保存直接使用测得dz的Spearman作敏感性对照。不提供把配对方向当独立样本的p值。','',
               '## 相关性与跨状态稳定性','',
               f'总体描述性Spearman={summary["pooled_spearman_radius_ties"]:.4f}。逐状态rho中位数={np.median(valid_rhos):.4f}，q05={np.quantile(valid_rhos,.05):.4f}，q95={np.quantile(valid_rhos,.95):.4f}，min={valid_rhos.min():.4f}，max={valid_rhos.max():.4f}。',
               f'逐状态rho≥0.9的比例={summary["fraction_states_rho_ge_09"]:.1%}，rho≥0.8的比例={summary["fraction_states_rho_ge_08"]:.1%}。有效rho状态数={len(valid_rhos)}。',
               f'沿同一方向逐级增大半径时，非减的相邻比较比例={summary["adjacent_radius_nondecreasing_fraction"]:.1%}；全部半径均非减的方向比例={summary["all_radii_nondecreasing_direction_fraction"]:.1%}。','',
               '## 各半径动作距离与增益分布','',
               '| r | da均值±标准差 | da q05/q50/q95 | gain q05/q50/q95 | gain q95/q05 | r/||z0||中位数 |','|---:|---|---|---|---:|---:|']
        for r in radius_records:lines.append(f'| {r["radius"]:g} | {r["action_distance_mean"]:.4f} ± {r["action_distance_std"]:.4f} | {r["action_distance_q05"]:.4f}/{r["action_distance_q50"]:.4f}/{r["action_distance_q95"]:.4f} | {r["gain_q05"]:.4f}/{r["gain_q50"]:.4f}/{r["gain_q95"]:.4f} | {r["gain_q95_over_q05"]:.2f} | {r["median_radius_over_z0_norm"]:.3f} |')
        lines += ['', '表中每个半径合并不同状态和方向；判断纯方向差异应读state_radius_statistics.csv，而不能只读合并分布。',
                  'state_radius_statistics.csv对每个状态/半径分别给出均值、样本标准差、q05/q25/q50/q75/q95、CV、q95/q05和max/min，包含da和gain，以及裁剪前对照。',
                  'state_statistics.csv记录每个状态的rho、配对方向单调性及参考半径的gain分布；radius_statistics.csv还记录不同状态的中位gain分布。',
                  '半径大小结合z0模长、r/||z0||和每坐标RMS=r/√28一起解释。r=2不是微小扰动；大半径结果可能受非线性与裁剪影响。',
                  '高Spearman只表示这些采样半径下的排序关系，不能证明等距映射、方向无关、距离比例固定或全局Lipschitz界；20个方向的最大最小值也不是Jacobian奇异值。','',
                  '[总览](distance_overview.png)；[逐状态散点示例](state_examples.png)；[同方向半径扫描](direction_rays.png)。',
                  'distance_samples.npz保存全部原始状态、动作、z0、单位方向、扰动z和动作、距离与增益；samples_long.csv每行一个扰动点。','',
                  '复现：','```bash',f'.venv/bin/python scripts/dsrlna/analyze_demo_latents.py --latent-action-distance-experiment --steps {step} --distance-num-states {B} --distance-num-directions {M} --distance-radii '+' '.join(f'{r:g}' for r in radii)+f' --batch-size {args.batch_size} --platform {args.platform} --seed {args.seed}','```','']
        (out/'REPORT.md').write_text('\n'.join(lines))
        print(json.dumps(dict(pooled_rho=summary['pooled_spearman_radius_ties'],state_rho=summary['per_state_spearman_summary'],monotone_fraction=summary['all_radii_nondecreasing_direction_fraction'])),flush=True)
        print(f'Latent/action distance results: {out}',flush=True)


def analyze_demo_flow_geometry(args, run, config, paths):
    """Run all four parts of experiment 4 with the shared inverse and decoder."""
    from scripts.dsrlna.demo_flow_geometry import run_experiment
    return run_experiment(args, run, config, paths, load_demos, make_functions)


def main():
    args = parse_args()
    if args.geometry_postprocess_from:
        from scripts.dsrlna.demo_flow_geometry import postprocess
        postprocess(args.geometry_postprocess_from, args.geometry_bootstrap)
        return
    if args.jacobian_minus_identity_from:
        analyze_jacobian_minus_identity(args.jacobian_minus_identity_from)
        return
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
    out = args.output / run.name / FLOW_INVERSE_OUTPUT_DIR
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
    if args.flow_geometry_experiment:
        analyze_demo_flow_geometry(args, run, config, paths)
        return
    if args.latent_action_distance_experiment:
        analyze_latent_action_distances(args, run, config, paths)
        return
    if args.latent_coordinate_experiment:
        analyze_latent_coordinate_interventions(args, run, config, paths)
        return
    if args.invert_all_shifted_times:
        analyze_all_demo_shifted_inverse(args, run, config, paths)
        return
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
    out = args.output / run.name / FLOW_INVERSE_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'demos.npz', **data)
    metadata = dict(run_dir=str(run), dataset=str(args.dataset.resolve()), available_full_chunks=available,
                    num_samples=n, horizon=horizon, seed=args.seed, num_draws=args.num_draws,
                    tolerance=args.tolerance, inverse_method=FLOW_INVERSE_METHOD,
                    reverse_query_times=list(inverse_time_grid(config['flow_steps'])),
                    reverse_times=list(inverse_time_grid(config['flow_steps'])),
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
