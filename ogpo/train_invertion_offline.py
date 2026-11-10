"""Offline teacher flow -> inverse Gaussian encoder -> freshly trained flow.

Example (inherits the original task's hyperparameters):
    bash scripts/offline/train_invertion.sh square --device=0 --seed=1 \
        --flow_checkpoint_path=checkpoints/offline/square-mh-low_dim/seed_1.pkl

Omit flow_checkpoint_path to first train the teacher exactly as train_offline.
The final flow uses DSRLResidual BC with original offline ODE and SDE/BON evaluation.
All stages share one online W&B run so a stage transition needs no network
handshake. Final training retains its own step axis starting at zero.
"""
import copy
import glob
import os
import pickle
import random
import time
from types import SimpleNamespace
from typing import Optional

if "CUDA_VISIBLE_DEVICES" in os.environ:
    _egl_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    os.environ["EGL_DEVICE_ID"] = _egl_device
    os.environ["MUJOCO_EGL_DEVICE_ID"] = _egl_device

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from ogpo.agents.ogpo import OGPOAgent
from ogpo.agents.dsrlresidual import DSRLResidualAgent
from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.config_loader import load_config, parse_cli_args, config_to_agent_dict
from ogpo.agents.modules.pg_helper import sample_flow_actions_sde
from ogpo.utils.flow_checkpoint import load_flow_checkpoint, save_flow_checkpoint, _checked_weights
from ogpo.utils.log_utils import make_run_timestamp
from ogpo.utils.main_helper import (
    make_vectorized_env, process_train_dataset, get_q_bounds,
    setup_experiment_logging,
)
from ogpo.utils.sharding import create_mesh, shard_agent

from ogpo.runners.offline_runner import run_offline_pretraining
from ogpo.runners.dsrlresidual_runner import run_residual_base_pretraining


class InvertionEvaluationAgent(OGPOAgent):
    """Keep OGPO's original SDE/BON evaluation, changing only its source."""
    base_policy_network: Optional[object] = None
    base_distribution = DSRLResidualAgent.base_distribution

    def sample_actions_with_noise(self, observations, grad_params=None, rng=None,
                                 use_target=True, is_encoded=False, images=None):
        # OGPO's inherited sample_actions/sample_actions_BON call this adapter.
        # The integration, diffusion schedule and clipping remain in the
        # original shared sampler; only the initial Gaussian is supplied here.
        params = grad_params if grad_params is not None else self.actor_network.params
        actor_fn = self.actor_network.select('target_actor')
        noise_fn = self.actor_network.select('target_noise_net')
        result = sample_flow_actions_sde(
            actor_fn=actor_fn, noise_fn=noise_fn, observations=observations, rng=rng,
            flow_steps=self.config.flow_steps, act_dim=self.config.noise_dim,
            use_constant_noise=self.config.use_constant_noise,
            use_tapered_noise=self.config.get('use_tapered_noise', False),
            error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
            constant_noise_std=self.config.constant_noise_std,
            min_noise_std=self.config.min_noise_std, randn_clip_value=self.config.randn_clip_value,
            act_min=self.config.act_min, act_max=self.config.act_max,
            clip_intermediate=self.config['clip_intermediate_actions'],
            clip_value=self.config['denoised_clip_value'], is_encoded=True, params=params,
            ft_flow_steps=self.config.get('ft_flow_steps', 0),
            base_dist=self.base_distribution(observations))
        actions, chains, logprob, sigmas = result
        ft = self.config.get('ft_flow_steps', 0)
        effective = ft if 0 < ft < self.config.flow_steps else self.config.flow_steps
        steps = effective + 1 if effective == self.config.flow_steps else effective
        if self.config.use_constant_noise:
            steps -= 1
        if self.config['normalize_denoising_horizon']:
            logprob = logprob / max(steps, 1)
        if self.config['normalize_act_space_dimension']:
            logprob = logprob / self.config.noise_dim
        return actions, chains, logprob, sigmas


class InvertionOfflineAgent(DSRLResidualAgent):
    """Residual flow training with original offline ODE and SDE/BON evaluation."""
    evaluation_template: Optional[OGPOAgent] = None

    @jax.jit
    def bc_update(self, batch):
        agent, info = super().bc_update(batch, use_base_distribution=True)
        # Match original offline evaluation: ODE uses the online decoder,
        # SDE/BON uses its EMA. Reuse OGPO's own EMA update implementation.
        template = agent.evaluation_template
        params = dict(template.actor_network.params)
        params['modules_actor'] = agent.bc_flow_network.params['modules_actor_bc_flow']
        state = template.actor_network.replace(params=params)
        template.target_update(state, 'actor')
        agent = agent.replace(evaluation_template=template.replace(actor_network=state))
        loss = info['bc_flow_loss']
        info = {key: value for key, value in info.items()
                if key not in ('bc_flow_loss', 'flow_uses_base_distribution')}
        return agent, dict(info, bc_loss=loss, bc_flow_loss_vel=loss,
                           bc_flow_loss_denoising=jnp.float32(0.0))

    def reset_flow_for_retraining(self, ex_observations, ex_actions):
        agent = super().reset_flow_for_retraining(ex_observations, ex_actions)
        template = agent.evaluation_template
        params = dict(template.actor_network.params)
        flow = agent.bc_flow_network.params['modules_actor_bc_flow']
        params['modules_actor'] = flow
        params['modules_target_actor'] = flow
        return agent.replace(evaluation_template=template.replace(
            actor_network=template.actor_network.replace(params=params)))

    @jax.jit
    def compute_flow_actions(self, observations, noises=None, params=None, rng=None):
        if noises is None:
            noises = self.base_distribution(observations).sample(seed=rng)
        if params is not None:
            return super().compute_flow_actions(observations, noises=noises, params=params, rng=rng)
        # Reuse the original offline ODE implementation. The outer JIT compiles
        # it; bypass its inner JIT wrapper because that marks noises as static.
        return OGPOAgent._compute_flow_actions_corrected_impl.__wrapped__(
            self._evaluation_policy(), observations, rng=rng, noises=noises,
            is_encoded=False, _add_batch_dim=(observations.ndim == 1))

    def _evaluation_policy(self):
        # Refresh the online decoder; keep the EMA updated by OGPO.target_update.
        # Keep OGPO's original noise network and action critic for SDE/BON.
        template = self.evaluation_template
        params = dict(template.actor_network.params)
        flow = self.bc_flow_network.params['modules_actor_bc_flow']
        params['modules_actor'] = flow
        config = template.config.copy()
        config['noise_dim'] = self.config['noise_dim']
        config['use_one_variance'] = self.config.get('use_one_variance', False)
        return InvertionEvaluationAgent(
            rng=self.rng, actor_network=template.actor_network.replace(params=params),
            critic_network=self.action_critic_network, config=config,
            base_policy_network=self.base_policy_network)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        return self._evaluation_policy().sample_actions(observations, rng=rng)

    @jax.jit
    def sample_actions_BON(self, observations, rng=None):
        return self._evaluation_policy().sample_actions_BON(observations, rng=rng)


def make_inverse_agent(teacher, example, config, seed):
    """Use residual's existing constructor; teacher flow is only for inversion."""
    residual_config = load_config('dsrlresidual')['agent']
    residual_config.update(dict(teacher.config))
    residual_config.update(
        agent_name='dsrlresidual', noise_chunk_length=-1,
        use_normal_base_encoder=False, use_one_variance=False,
        use_invertionencoder_retrainflow=False,
        bc_flow_hidden_dims=teacher.config['actor_hidden_dims'],
        lr_actor=float(config.get('lr_actor') or config.get('actor_lr') or config['lr']),
        lr_critic=float(config.get('lr_critic') or config.get('critic_lr') or config['lr']),
        lr_actor_bc_flow=float(config.get('lr_actor_bc_flow') or config.get('actor_lr') or config['lr']))
    agent = InvertionOfflineAgent.create(
        seed, example['observations'], example['actions'], residual_config)
    params = dict(agent.bc_flow_network.params)
    params['modules_actor_bc_flow'] = _checked_weights(
        params['modules_actor_bc_flow'],
        teacher.actor_network.params['modules_target_actor'], 'inverse teacher')
    return agent.replace(
        bc_flow_network=agent.bc_flow_network.replace(params=params),
        action_critic_network=teacher.critic_network, evaluation_template=teacher)


def train_inverse_encoder(agent, dataset, config, logger, start_step, mesh=None):
    """Run residual's original inversion and encoder-BC loops, without retraining yet."""
    encoder_logger = SimpleNamespace(log=lambda info, prefix, step:
        logger.log(info, 'inverse_encoder', step=step))
    encoder_config = dict(config, use_normal_base_encoder=False,
                          use_invertionencoder_retrainflow=False)
    agent, _ = run_residual_base_pretraining(
        agent, dataset, encoder_config, encoder_logger,
        start_step=start_step, mesh=mesh)
    return agent


class _StageWandbLogger:
    """Keep W&B event steps increasing while each stage has a local x-axis."""

    def __init__(self, backend, stage, offset, prefixes):
        self.backend = backend
        self.stage = stage
        self.offset = offset
        self.step_metric = f'{stage}/step'
        backend.define_metric(self.step_metric)
        for prefix in prefixes:
            metric_prefix = f'prepare/{prefix}' if stage == 'prepare' else prefix
            backend.define_metric(
                f'{metric_prefix}/*', step_metric=self.step_metric, step_sync=False)

    def log(self, data, step=None, **kwargs):
        if self.stage == 'prepare':
            data = {f'prepare/{key}': value for key, value in data.items()}
        else:
            data = dict(data)
        if step is not None:
            data[self.step_metric] = step
            kwargs['step'] = self.offset + step
        self.backend.log(data, **kwargs)


def _start_logging(config, stage):
    exp = config['experiment']
    stage_config = copy.deepcopy(config)
    stage_config['invertion_stage'] = stage
    stage_config['flow_source_distribution'] = 'inverse_encoder' if stage == 'retrain' else 'normal'
    if stage == 'retrain':
        stage_config['evaluation_sampler'] = 'offline_ode_and_sde_bon'
    result = setup_experiment_logging(
        seed=exp['seed'], project=exp['project'], run_group=exp['run_group'],
        wandb_name=exp['wandb_name'] + ('_invertion_offline' if stage == 'retrain' else '_invertion_teacher'),
        save_dir=exp['save_dir'], env_name=config['env']['name'],
        log_enabled=exp['log'],
        prefixes=['eval', 'eval_sde', 'offline_agent', 'memory', 'step_data', 'inverse_encoder'],
        run_timestamp=make_run_timestamp(), config=stage_config,
        reuse_wandb_run=stage == 'retrain')
    _, logger, _ = result
    if logger.wandb_logger is not None:
        logger.wandb_logger = _StageWandbLogger(
            wandb, stage, wandb.run.step, logger.csv_loggers)
    return result


def _finish_logging(logger):
    logger.close()
    if logger.log_enabled:
        wandb.finish()


def _train_flow(agent, dataset, eval_env, eval_envs, config, logger, save_dir, mesh=None):
    """Use the same runner, metrics, evaluation and stopping as train_offline."""
    pending_eval = {}
    def log_metrics(data, prefix, step):
        # Preserve native eval_sde/* and also expose the explicitly requested
        # eval/success_sde beside eval/success, in the same evaluation row.
        if prefix == 'eval':
            pending_eval[step] = dict(data)
            return
        logger.log(data, prefix, step=step)
        if prefix == 'eval_sde' and step in pending_eval:
            ode_data = pending_eval.pop(step)
            ode_data['success_sde'] = data.get('success', 0.0)
            logger.log(ode_data, 'eval', step=step)
    result = run_offline_pretraining(
        agent=agent, train_dataset=dataset, eval_env=eval_env, eval_envs=eval_envs,
        config=config, agent_config=agent.config,
        logger=SimpleNamespace(log=log_metrics),
        save_dir=save_dir, log_step=0, mesh=mesh)
    for step, data in pending_eval.items():
        logger.log(data, 'eval', step=step)
    return result


def _parse_launch_args():
    """Merge scoped Bash options, keeping direct Python CLI overrides first."""
    algo, overrides = parse_cli_args()
    for name, key in (
            ('OGPO_INVERTION_BASE_BC_STEPS', 'agent.base_bc_steps'),
            ('OGPO_INVERTION_TRAINING_STEP', 'training_step'),
            ('OGPO_INVERTION_EARLY_STOP', 'early_stop'),
            ('OGPO_INVERTION_USE_ONE_VARIENCE', 'use_one_varience'),
            ('OGPO_INVERTION_USE_OT_OPTIMIZATION', 'use_OT_optimization')):
        if name in os.environ and key not in overrides:
            _, extra = parse_cli_args([f'--{key}={os.environ[name]}'])
            overrides.update(extra)
    return algo, overrides


def main():
    algo, overrides = _parse_launch_args()
    if algo != 'ogpo':
        raise ValueError('train_invertion_offline requires --algo=ogpo')
    config = load_config(algo, overrides)
    config['training'].update(bc_only=True, online_steps=0, bc_q_steps=0, calql_steps=0)
    ac = config['agent']
    ac.setdefault('base_bc_steps', 100000)
    if config['training']['bc_pi_steps'] <= 0 or ac['base_bc_steps'] <= 0:
        raise ValueError('training.bc_pi_steps and agent.base_bc_steps must be positive')
    training_step = config.setdefault('training_step', 90000)
    early_stop = config.setdefault('early_stop', False)
    use_one_varience = config.setdefault('use_one_varience', False)
    use_OT_optimization = config.setdefault('use_OT_optimization', False)
    if isinstance(training_step, bool) or not isinstance(training_step, int) or training_step <= 0:
        raise ValueError('training_step must be a positive integer')
    if not isinstance(early_stop, bool):
        raise ValueError('early_stop must be true or false')
    if not isinstance(use_one_varience, bool):
        raise ValueError('use_one_varience must be true or false')
    if not isinstance(use_OT_optimization, bool):
        raise ValueError('use_OT_optimization must be true or false')
    ckpt = config['checkpoint']
    if any(ckpt.get(key) for key in ('resume_path', 'restore_actor_path', 'restore_critic_path')):
        raise ValueError('Use checkpoint.flow_path for the teacher; online resume/individual restores are unsupported')
    if ac.get('policy_type', 'flow') != 'flow' or any(ac.get(k, False) for k in (
            'use_shortcut', 'use_denoiser', 'use_one_step_policy', 'useSimBa', 'fpo_zero_sampling')):
        raise ValueError('This comparison requires standard velocity flow matching without shortcut, denoiser, one-step, SimBa, or zero sampling')
    if ac.get('actor_obs', 'state') != 'state' or ac.get('critic_obs', 'state') != 'state' or ac.get('freeze_vision_encoder', False):
        raise ValueError('This standalone inverse-encoder comparison currently requires state observations (low_dim)')
    exp, ec, dc = (config[k] for k in ('experiment', 'env', 'dataset'))
    seed, env_name = exp['seed'], ec['name']
    random.seed(seed)
    np.random.seed(seed)
    if dc['ogbench_dir'] is not None:
        paths = [p for p in sorted(glob.glob(f"{dc['ogbench_dir']}/*.npz")) if '-val.npz' not in p]
        if not paths:
            raise FileNotFoundError(f"No training datasets in {dc['ogbench_dir']}")
        env, eval_env, dataset, val_dataset = make_ogbench_env_and_datasets(
            env_name, dataset_path=paths[0], compact_dataset=False)
    else:
        env, eval_env, dataset, val_dataset = make_env_and_datasets(
            env_name, rew_fn=ec['rew_fn'], post_success_steps=ec.get('post_success_steps', 0))
    del val_dataset
    eval_envs = None
    if config['eval']['n_eval_envs'] > 1:
        eval_envs = make_vectorized_env(
            env_name, num_envs=config['eval']['n_eval_envs'], seed=seed + 1000,
            rew_fn=ec['rew_fn'], post_success_steps=ec.get('post_success_steps', 0))
    agent_config = config_to_agent_dict(config)
    agent_config['q_min'], agent_config['q_max'] = get_q_bounds(env_name, agent_config['discount'])
    dataset = process_train_dataset(
        dataset, env_name, dataset_proportion=dc['proportion'],
        num_offline_trajs=dc.get('num_offline_trajs', -1), sparse=ec['sparse'],
        numpy_rng=np.random.default_rng(seed + 20000), compute_mc_returns=True,
        discount=agent_config['discount'])
    dataset.p_aug = dc['p_aug']
    example = dataset.sample(())
    if 'images' in example:
        raise ValueError('Use a low_dim dataset for this script')
    def create_agent():
        return OGPOAgent.create(
            seed, example['observations'], example['actions'], agent_config,
            agent_config['adv_clip_min'])
    dataset_rng_state = copy.deepcopy(dataset.numpy_rng.bit_generator.state)
    dataset_jax_rng = dataset.rng
    agent = create_agent()
    mesh = create_mesh(config['training'].get('gpu_mode', 'single'))
    if mesh is not None:
        if agent_config['batch_size'] % len(mesh.devices):
            raise ValueError('batch_size must be divisible by the number of mesh devices')
        agent = shard_agent(agent, mesh)
    save_dir, logger, _ = _start_logging(config, 'prepare')
    try:
        teacher_steps = 0
        if ckpt.get('flow_path'):
            agent = load_flow_checkpoint(agent, ckpt['flow_path'], env_name=env_name)
            if mesh is not None:
                agent = shard_agent(agent, mesh)
            print(f"Loaded teacher flow from {ckpt['flow_path']}; skipping normal-source BC")
        else:
            agent, teacher_steps, _ = _train_flow(
                agent, dataset, eval_env, eval_envs, config, logger, save_dir, mesh)
            save_flow_checkpoint(agent, os.path.join(save_dir, 'teacher_flow.pkl'),
                                 env_name=env_name, seed=seed, step=teacher_steps)
        # Switch to residual's own flow/encoder implementation before inversion.
        agent = make_inverse_agent(agent, example, agent_config, seed)
        if mesh is not None:
            agent = shard_agent(agent, mesh)
        agent = train_inverse_encoder(agent, dataset, agent_config, logger, teacher_steps, mesh)
        save_agent(agent, save_dir, teacher_steps + agent_config['base_bc_steps'])
        # Flush the preparation row locally, keeping the online run alive.
        logger.close()
        # Exactly the same decoder reset as use_invertionencoder_retrainflow:
        # fit the encoder first, then reinitialize only the flow and its optimizer.
        dataset.numpy_rng.bit_generator.state = dataset_rng_state
        dataset.rng = dataset_jax_rng
        random.seed(seed)
        np.random.seed(seed)
        agent = agent.replace(config=agent.config.copy(
            dict(use_one_variance=use_one_varience,
                 use_OT_optimization=use_OT_optimization)))
        agent = agent.reset_flow_for_retraining(example['observations'], example['actions'])
        if mesh is not None:
            agent = shard_agent(agent, mesh)
        retrain_config = copy.deepcopy(config)
        retrain_config['training']['bc_pi_steps'] = training_step
        retrain_config['bc']['clip_bc'] = early_stop
        retrain_config['agent']['use_one_variance'] = use_one_varience
        retrain_config['agent']['use_OT_optimization'] = use_OT_optimization
        save_dir, logger, _ = _start_logging(retrain_config, 'retrain')
        started = time.monotonic()
        agent, step, info = _train_flow(
            agent, dataset, eval_env, eval_envs, retrain_config, logger, save_dir, mesh)
        save_agent(agent, save_dir, step)
        output = ckpt.get('export_flow_path') or os.path.join(
            'checkpoints', 'invertion_offline', env_name, f'seed_{seed}.pkl')
        # A learned-source decoder needs BOTH flow and source parameters. Use a
        # distinct format so standard-normal loaders cannot silently misuse it.
        payload = dict(
            format='ogpo_invertion_offline_v1', env_name=env_name, seed=seed,
            bc_step=step, config=dict(agent.config),
            flow_module='modules_actor_bc_flow',
            flow_params=jax.device_get(agent.bc_flow_network.params),
            base_encoder_params=jax.device_get(agent.base_policy_network.params))
        output = os.path.abspath(os.path.expanduser(output))
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = output + '.tmp'
        with open(temporary, 'wb') as stream:
            pickle.dump(payload, stream)
        os.replace(temporary, output)
        print(f"Saved encoder + retrained flow to {output}")
        print(f"Flow retraining: {step} updates in {time.monotonic() - started:.1f}s; success={info.get('success', 'N/A')}")
    finally:
        _finish_logging(logger)
        for environment in (env, eval_env, eval_envs):
            if environment is not None:
                environment.close()


if __name__ == '__main__':
    main()
