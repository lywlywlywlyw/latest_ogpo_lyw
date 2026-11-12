"""Offline inverse-latent base BC and online DSRLResidual training.

Retains DSRL-NA chunk replay, critic updates and split UTD scheduling.
"""

from types import SimpleNamespace
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.evaluation import evaluate, evaluate_parallel, _split_obs
from ogpo.utils.main_helper import LoggingHelper, save_rolling_checkpoint
from ogpo.utils.sharding import shard_batch
from ogpo.utils.mem_monitor import log_memory
from ogpo.runners.bc_runner import _log_step_data
from ogpo.runners.online_rl_runner import (
    _wrap_actor_for_frozen_encoder, _state_for_encoder,
)


DSRL_ALGORITHMS = {'dsrlresidual'}


def _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name):
    if n_eval_envs <= 1:
        return None
    return SimpleNamespace(
        n_eval_envs=n_eval_envs, eval_episodes=eval_episodes,
        discount=discount, run_name=run_name, env_name=env_name,
    )


def _run_evaluation(agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot, encode_fn=None):
    if eval_flags is None:
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, global_step=log_step,
            action_dim=action_dim, num_eval_episodes=eval_episodes,
            num_video_episodes=video_episodes, video_frame_skip=video_frame_skip,
            actor_fn=actor_fn, env_name=run_name, plot=plot, encode_fn=encode_fn,
        )
    else:
        eval_info = evaluate_parallel(
            eval_envs, agent, eval_flags, action_dim, log_step,
            actor_fn=actor_fn, plot=plot, encode_fn=encode_fn,
        )
    return eval_info


def _reshape_for_utd(batch, utd_ratio, batch_size):
    return jax.tree.map(lambda x: x.reshape((utd_ratio, batch_size) + x.shape[1:]), batch)


def invert_demo_latents(agent, train_dataset, config):
    """Reverse full action chunks, padding at episode boundaries."""
    batch_size = config['batch_size']
    # Agent creation resolves dimensions on a config copy; the input may still contain None.
    horizon = config['horizon_length'] if config['action_chunking'] else 1
    ends = np.unique(np.append(np.flatnonzero(train_dataset['terminals']), train_dataset.size - 1))
    latents = np.empty((train_dataset.size, agent.config['noise_dim']), dtype=np.float32)
    for start in tqdm.trange(0, train_dataset.size, batch_size, desc='Invert demos for base BC'):
        idxs = np.arange(start, min(start + batch_size, train_dataset.size))
        last = ends[np.searchsorted(ends, idxs)]
        # Never cross episodes; repeat the terminal action in a short chunk.
        seq = np.minimum(idxs[:, None] + np.arange(horizon), last[:, None])
        actions = train_dataset['actions'][seq].reshape(len(idxs), -1)
        observations = train_dataset.sample(len(idxs), idxs=idxs)['observations']
        latents[idxs] = np.asarray(agent.actions_to_noise(observations, actions))
    if not np.isfinite(latents).all():
        raise ValueError('Nonfinite inverted demo latents')
    return latents


def run_residual_base_pretraining(agent, train_dataset, config, logger, start_step=0, mesh=None,
                                 env_name=None, eval_env=None, eval_envs=None, save_dir=None,
                                 n_eval_envs=32, eval_episodes=2, log_interval=1000,
                                 step_data_log_interval=10):
    """After flow BC: invert demos and fit the base, unless using N(0, I)."""
    if config.get('use_normal_base_encoder', False):
        print('Using standard normal base N(0,I); skipping demo inversion and base encoder training')
        return agent, start_step
    from ogpo.utils.sharding import shard_batch_flat
    batch_size = config['batch_size']
    horizon = config['horizon_length'] if config['action_chunking'] else 1
    latents = invert_demo_latents(agent, train_dataset, config)
    steps = config['base_bc_steps']
    if steps < 1:
        raise ValueError('base_bc_steps must be positive for a fresh DSRLResidual run')
    for i in tqdm.trange(1, steps + 1, desc='Base Gaussian BC'):
        idxs = train_dataset.get_random_idxs(batch_size)
        batch = dict(observations=train_dataset.sample(batch_size, idxs=idxs)['observations'],
                     actions=latents[idxs])
        if mesh is not None:
            batch = shard_batch_flat(batch, mesh)
        agent, info = agent.base_bc_update(batch)
        if i % 1000 == 0 or i == steps:
            logger.log(info, 'offline_agent', step=start_step + i)
    log_step = start_step + steps
    if config.get('use_invertionencoder_retrainflow', False):
        # Inversion must use the loaded teacher. Only now replace its weights;
        # the fitted base, residual actor and critics remain intact.
        example = train_dataset.sample(())
        agent = agent.reset_flow_for_retraining(example['observations'], example['actions'])
        if mesh is not None:
            from ogpo.utils.sharding import shard_agent
            agent = shard_agent(agent, mesh)
        print('Flow retraining: random initialization with a fresh optimizer (checkpoint used only for inversion)')
        # Same task defaults as scripts/offline/train.sh -> scripts/ogpo/*.sh.
        maximum, eval_interval, early_stop = {
            'square': (500000, 5000, True),
            'transport': (1000000, 20000, False),
            'tool_hang': (500000, 10000, True),
        }[env_name.split('-')[0]]
        retrain_steps = config.get('invertionencoder_retrainflow_steps') or maximum
        eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, config['discount'], env_name, env_name)
        memory_interval = max(eval_interval, 50000)
        log_memory('flow_retrain_start', step=log_step, logger=logger)
        for i in tqdm.trange(1, retrain_steps + 1, desc='Flow retraining from frozen base encoder'):
            batch = train_dataset.sample_sequence(
                batch_size, sequence_length=horizon, discount=config['discount'])
            if mesh is not None:
                batch = shard_batch_flat(batch, mesh)
            agent, info = agent.bc_update(batch, use_base_distribution=True)
            log_step += 1
            if step_data_log_interval > 0 and i % step_data_log_interval == 0:
                _log_step_data(logger, batch, 'flow_retrain', log_step, prefix='offline_step_data')
            should_log = (i == 1 or i == retrain_steps or i % eval_interval == 0 or
                          (log_interval > 0 and i % log_interval == 0))
            if should_log:
                logger.log(dict(info, retrain_step=i), 'offline_flow', step=log_step)
            if i % eval_interval == 0 or i == retrain_steps:
                eval_info = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    partial(agent.sample_base_actions, num_samples=8), log_step,
                    eval_episodes, 0, 3, env_name, plot=False)
                logger.log(dict(eval_info, retrain_step=i), 'offline_eval', step=log_step)
                success = eval_info.get('success', 0.)
                print(f'Flow retraining step={i}: Best-of-8 success={success:.4f}')
                if early_stop and success >= 0.5:
                    save_agent(agent, save_dir, log_step)
                    break
            if i % memory_interval == 0:
                log_memory('flow_retrain', step=log_step, logger=logger)
        log_memory('flow_retrain_end', step=log_step, logger=logger)
    return agent, log_step


def run_dsrlresidual_online_rl(
    agent: Any,
    env: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    train_dataset: Dataset,
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    algo_name: str,
    online_steps: int,
    start_training: int,
    batch_size: int,
    horizon_length: int,
    discount: float,
    utd_warmup: int = 1,
    utd_online: int = 1,
    utd_q: Optional[int] = None,
    utd_pi: Optional[int] = None,
    utd_v: Optional[int] = None,
    offline_ratio: float = 0.0,
    bc_refine_steps: int = 0,
    q_warmup_steps: int = 0,
    use_success_buffer: bool = False,
    success_buffer_batch_size: int = 256,
    best_of_n: int = 1,
    eval_interval: int = 100000,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'dsrlresidual',
    env_name: str = '',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    save_interval: int = 500000,
    preemption_backup_interval: int = -1,
    checkpoint_dir: Optional[str] = None,
    log_enabled: bool = True,
    sparse: bool = False,
    p_aug: Optional[float] = None,
    buffer_size: int = 2000000,
    seed: int = 0,
    is_resumed: bool = False,
    resumed_online_step: int = 0,
    resumed_replay_buffer: Optional[ReplayBuffer] = None,
    start_step: int = 0,
    wandb_run_id: Optional[str] = None,
    force_jax_sync: bool = True,
    encode_fn: Optional[Any] = None,
    mc_regression: bool = False,
    mesh=None,
) -> Tuple[Any, int, Dict, ReplayBuffer]:
    """Online RL runner for DSRLResidual. Stores chunk-level transitions with
    noise as 'actions' and denoised actions as 'true_actions'."""
    if use_success_buffer or bc_refine_steps:
        raise ValueError('DSRLResidual uses a mean-residual actor with optional KL and a frozen decoder')
    assert algo_name in DSRL_ALGORITHMS, f"Expected DSRL algorithm, got {algo_name}"
    assert offline_ratio == 0.0, f"DSRL requires offline_ratio=0.0, got {offline_ratio}"

    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]

    noise_chunk_length = config.get('noise_chunk_length', -1)
    if noise_chunk_length == -1:
        noise_dim = action_dim * horizon_length
    else:
        noise_dim = action_dim * noise_chunk_length

    is_dsrl_expo = (algo_name == 'dsrl_plus_expo')

    if is_resumed and resumed_replay_buffer is not None:
        replay_buffer = resumed_replay_buffer
    else:
        example_noise = np.zeros((noise_dim,))
        example_true_actions = np.zeros((action_dim * horizon_length,))

        example_transition = dict(
            observations=example_batch['observations'],
            actions=example_noise,
            true_actions=example_true_actions,
            rewards=np.array(0.0),
            discounts=np.array(discount ** horizon_length),
            chunk_steps=np.array(horizon_length, dtype=np.int32),
            terminals=np.array(0.0),
            masks=np.array(1.0),
            next_observations=example_batch['observations'],
        )
        if is_dsrl_expo:
            example_transition['refined_actions'] = example_true_actions.copy()
            example_transition['base_actions'] = example_true_actions.copy()

        replay_buffer = ReplayBuffer.create(
            example_transition, buffer_size,
            numpy_rng=np.random.default_rng(seed + 30000))

    if p_aug is not None:
        replay_buffer.p_aug = p_aug

    def _maybe_shard(b):
        return shard_batch(b, mesh) if (mesh is not None and b is not None) else b

    # Frozen-encoder plumbing: image envs return dict obs {state, image, full_state}.
    # The agent was init'd on the encoded [img_feats, proprio] vector, so every env
    # observation must be encoded the same way before being passed to the agent.
    _use_full_state = config.get('use_state', 'proprio') == 'full'
    _actor_uses_state = config.get('_original_actor_obs', 'state') == 'state'

    def _encode_ob(raw_ob):
        """Split env dict obs and fold image features into a flat encoded vector.

        For low-dim envs (flat obs) this is a no-op pass-through.
        """
        ob_split, ob_images, ob_full_state = _split_obs(raw_ob)
        if encode_fn is not None and ob_images is not None:
            return np.asarray(encode_fn(
                _state_for_encoder(ob_split, ob_full_state, _use_full_state)[None],
                ob_images[None],
            )[0])
        return ob_split

    raw_ob, _ = env.reset(seed=seed + 10000)
    ob = _encode_ob(raw_ob)
    action_queue = []
    trajectory_buffer = []
    current_noise = None
    current_true_actions = None
    current_base_actions = None

    if not is_resumed and hasattr(agent, 'reset_optimizers_with_lr'):
        agent = agent.reset_optimizers_with_lr()
        print("Reset optimizer with online learning rate")

    log_step = start_step
    total_env_step = 0
    eval_step = 0
    next_eval_step = ((total_env_step // eval_interval + 1) * eval_interval
                      if eval_interval > 0 else float("inf"))
    next_save_step = save_interval if save_interval > 0 else float("inf")
    next_backup_step = (preemption_backup_interval if preemption_backup_interval > 0
                        else float("inf"))
    update_info = {}
    online_rng = jax.random.PRNGKey(seed + 40000)
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    # Same split-UTD convention as online_rl_runner: joint then critic-only.
    eff_utd_pi = utd_pi if utd_pi is not None else utd_online
    eff_utd_q = utd_q if utd_q is not None else utd_online
    if any(isinstance(u, bool) or not isinstance(u, int) or u < 1
           for u in (eff_utd_pi, eff_utd_q)) or eff_utd_q < eff_utd_pi:
        raise ValueError('UTD requires integer utd_q >= utd_pi >= 1')
    split_utd = eff_utd_q > eff_utd_pi
    if split_utd and not hasattr(agent, 'batch_critic_only_update'):
        raise ValueError(f'{algo_name} does not support separate utd_q and utd_pi')
    utd_ratio = eff_utd_q
    eval_info = {}

    if is_resumed and resumed_online_step > 0:
        total_env_step = resumed_online_step
        if save_interval > 0:
            next_save_step = (total_env_step // save_interval + 1) * save_interval
        if preemption_backup_interval > 0:
            next_backup_step = ((total_env_step // preemption_backup_interval + 1)
                                * preemption_backup_interval)
        print(f"Resuming DSRL online loop at env step {total_env_step}")
        next_eval_step = ((total_env_step // eval_interval + 1) * eval_interval
                          if eval_interval > 0 else float("inf"))
        if "discounts" not in replay_buffer._dict:
            raise ValueError("Legacy DSRL replay lacks actual chunk discounts; start a fresh run")

    next_log_step = ((total_env_step // log_interval + 1) * log_interval
                     if log_interval > 0 else float("inf"))
    logger.start_online_phase(log_step, online_step=total_env_step)

    with tqdm.tqdm(total=online_steps, initial=total_env_step, desc="DSRLResidual Online RL", dynamic_ncols=True) as pbar:
        while total_env_step < online_steps:
            online_rng, key = jax.random.split(online_rng)

            if len(action_queue) == 0:
                noise = agent.sample_noise(observations=ob, rng=key)
                current_noise = np.array(noise).flatten()
                action = agent.noise_to_actions_public(ob, noise)

                current_true_actions = np.array(action).flatten()
                action_chunk = np.array(action).reshape(-1, action_dim)
                for a in action_chunk:
                    action_queue.append(a)

            done_all = []
            reward_all = []
            terminated_all = []

            for t in range(len(action_queue)):
                if total_env_step >= online_steps:
                    break
                log_step += 1
                a = action_queue.pop(0)
                raw_next_ob, int_reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                done_all.append(done)
                reward_all.append(int_reward)
                terminated_all.append(terminated)
                total_env_step += 1
                pbar.update(1)
                if done:
                    break

            # Encode the post-chunk obs once for both the transition and next step
            next_ob = _encode_ob(raw_next_ob)

            chunk_steps = len(reward_all)
            reward = sum((discount ** t) * r for t, r in enumerate(reward_all))
            chunk_discount = discount ** chunk_steps
            done = any(done_all)
            terminated = any(terminated_all)

            env_info = {k: v for k, v in info.items() if k.startswith("distance")}
            env_info['total_env_steps'] = total_env_step
            try:
                logger.log(env_info, "env", step=log_step)
            except Exception:
                pass

            transition = dict(
                observations=ob,
                actions=current_noise,
                true_actions=current_true_actions,
                rewards=reward,
                discounts=chunk_discount,
                chunk_steps=chunk_steps,
                terminals=float(done),
                masks=1.0 - float(terminated),
                next_observations=next_ob,
            )
            if is_dsrl_expo:
                transition['refined_actions'] = current_true_actions
                transition['base_actions'] = current_base_actions

            trajectory_buffer.append(transition)

            if done or total_env_step >= online_steps:
                is_success = float(info.get('success', 0.0))
                score = is_success if is_success != 0 else 1e-9
                for traj_transition in trajectory_buffer:
                    if use_success_buffer:
                        traj_transition['is_success'] = is_success
                        traj_transition['score'] = score
                    replay_buffer.add_transition(traj_transition)

                trajectory_buffer = []
                if total_env_step < online_steps:
                    raw_ob, _ = env.reset(seed=seed + 10000 + total_env_step)
                    ob = _encode_ob(raw_ob)
                action_queue = []
                current_noise = None
                current_true_actions = None
                current_base_actions = None
            else:
                ob = next_ob

            # Extra critic updates must not delay the existing actor warmup gate.
            min_required = batch_size * eff_utd_pi
            if total_env_step >= start_training and replay_buffer.size >= min_required:
                batch = replay_buffer.sample(batch_size * utd_ratio)
                batch = _reshape_for_utd(batch, utd_ratio, batch_size)

                batch_success = None
                success_flag = False
                if use_success_buffer:
                    num_successful = replay_buffer._traj_success_mask[:replay_buffer.size].sum()
                    min_success_required = success_buffer_batch_size * eff_utd_pi
                    if num_successful >= min_success_required:
                        try:
                            batch_success = replay_buffer.sample(
                                success_buffer_batch_size * eff_utd_pi,
                                success_only=True)
                            batch_success = _reshape_for_utd(
                                batch_success, eff_utd_pi, success_buffer_batch_size)
                            success_flag = True
                        except ValueError:
                            success_flag = False

                if split_utd:
                    joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch)
                    agent, update_info["online_agent"] = agent.batch_update(
                        _maybe_shard(joint), _maybe_shard(batch_success), success_flag)
                    extra = jax.tree.map(lambda x: x[eff_utd_pi:], batch)
                    agent, critic_info = agent.batch_critic_only_update(_maybe_shard(extra))
                    update_info["online_agent"].update(critic_info)
                else:
                    agent, update_info["online_agent"] = agent.batch_update(
                        _maybe_shard(batch), _maybe_shard(batch_success), success_flag)
            if force_jax_sync:
                jax.block_until_ready(agent)
            if log_interval > 0 and update_info and (
                    total_env_step >= next_log_step or total_env_step >= online_steps):
                for key, info_dict in update_info.items():
                    logger.log(info_dict, key, step=log_step)
                update_info = {}
                next_log_step = (total_env_step // log_interval + 1) * log_interval

            if total_env_step >= online_steps or total_env_step >= next_eval_step:
                if eval_interval > 0:
                    next_eval_step = (total_env_step // eval_interval + 1) * eval_interval
                eval_step += 1
                # Wrap the actor with the frozen encoder so the eval loop's raw
                # dict obs is folded into the encoded vector the agent was init'd
                # against, else apply raises ScopeParamShapeError.
                eval_actor_fn = agent.sample_actions
                if encode_fn is not None:
                    eval_actor_fn = _wrap_actor_for_frozen_encoder(
                        eval_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state,
                        use_full_state=_use_full_state)
                eval_info = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    eval_actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name,
                    plot=(eval_step % 2 == 0) and plot_q_vs_mc,
                    encode_fn=encode_fn,
                )
                print(f"[EVAL] env_step={total_env_step} | "
                      f"success={float(eval_info.get('success', 0.0)):.4f}")
                logger.log(eval_info, "eval", step=log_step)

            # Chunks can cross a save boundary without landing on its exact step.
            if log_enabled and total_env_step >= next_save_step:
                save_agent(agent, save_dir, log_step)
                next_save_step = (total_env_step // save_interval + 1) * save_interval

            if total_env_step >= next_backup_step:
                next_backup_step = ((total_env_step // preemption_backup_interval + 1)
                                    * preemption_backup_interval)
                print(f"Saving rolling checkpoint at env step {total_env_step}")
                save_rolling_checkpoint(
                    agent, replay_buffer, log_step, total_env_step, seed, env_name,
                    wandb_run_id=wandb_run_id, algo_name=algo_name, checkpoint_dir=checkpoint_dir)

    if preemption_backup_interval > 0:
        save_rolling_checkpoint(agent, replay_buffer, log_step, total_env_step, seed, env_name,
                                wandb_run_id=wandb_run_id, algo_name=algo_name,
                                checkpoint_dir=checkpoint_dir)
    save_agent(agent, save_dir, log_step)
    print(f"DSRL online RL training complete. Final log_step: {log_step}")
    return agent, log_step, eval_info, replay_buffer
