"""Shared helpers for training pipelines: env creation, dataset processing,
checkpointing, logging, and Q-value bounds."""

import glob
import json
import os
import pickle
import random
import shutil
import time
import tempfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb, make_run_timestamp, timestamped_run_name


class LoggingHelper:
    """Helper class for unified logging to CSV and WandB."""

    def __init__(
        self,
        csv_loggers: Dict[str, CsvLogger],
        wandb_logger: Optional[Any] = None,
        log_enabled: bool = True,
    ):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.log_enabled = log_enabled
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data: Dict, prefix: str, step: int):
        """Log data to CSV and WandB."""
        if not self.log_enabled:
            return
        assert prefix in self.csv_loggers, f"Unknown prefix: {prefix}"
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger is not None:
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

    def close(self):
        """Close all CSV loggers."""
        for csv_logger in self.csv_loggers.values():
            if csv_logger is not None:
                csv_logger.close()


def get_env_maker(env_name: str) -> Callable:
    """Get the appropriate make_env function for the environment."""
    from envs.robomimic_utils import is_robomimic_env
    from envs.env_utils import GYMNASIUM_ROBOTICS_ADROITHAND_ENVS

    if is_robomimic_env(env_name):
        from envs.robomimic_utils import make_env
    elif env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        from envs.gymnasium_robotics_utils import make_env
    elif env_name.startswith('D4RL/kitchen'):
        from envs.franka_kitchen_utils import make_env
    elif env_name.startswith('stacking') or env_name.startswith('gate_insertion'):
        from envs.d3il_utils import make_env
    elif env_name.startswith('pusht'):
        from envs.pusht_utils import make_env
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return make_env


def make_vectorized_env(
    env_name: str,
    num_envs: int,
    seed: Optional[int] = None,
    rew_fn: str = 'sparse',
    post_success_steps: int = 0,
) -> AsyncVectorEnv:
    """Create vectorized environment."""
    make_env = get_env_maker(env_name)

    def env_fn(env_idx):
        if seed is None:
            env_seed = np.random.randint(0, 1000000)
        else:
            env_seed = seed + env_idx

        if env_name.startswith("pusht"):
            env = make_env(env_name, seed=env_seed, rew_fn=rew_fn)
        else:
            env = make_env(env_name, seed=env_seed, post_success_steps=post_success_steps)
        return env

    vec_env = AsyncVectorEnv(
        [lambda idx=i: env_fn(idx) for i in range(num_envs)],
        autoreset_mode=AutoresetMode.DISABLED
    )
    return vec_env


def get_checkpoint_dir(seed: int, env_name: str, algo_name: str = 'ogpo',
                       run_timestamp: Optional[str] = None) -> str:
    """Allocate a timestamped directory for one online experiment."""
    root = os.environ.get('OGPO_CHECKPOINT_ROOT', os.path.join(os.getcwd(), 'checkpoints'))
    return os.path.join(root, algo_name, f'{seed}_{env_name}',
                        run_timestamp or make_run_timestamp())


def save_rolling_checkpoint(
        agent: Any, replay_buffer: Any, step: int, online_loop_step: int,
        seed: int, env_name: str, wandb_run_id: Optional[str] = None,
        algo_name: str = 'ogpo', checkpoint_dir: Optional[str] = None,
    ):
    """Commit a complete step directory atomically; retain the latest two."""
    if checkpoint_dir is None:
        raise ValueError('Online checkpoint saving requires an explicit experiment directory')
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_dir = os.path.join(checkpoint_dir, str(step))
    if os.path.isdir(final_dir):
        if all(os.path.isfile(os.path.join(final_dir, name))
               for name in ('agent.pkl', 'buffer.pkl', 'meta.json')):
            return  # A final save may coincide with a periodic save.
        raise FileExistsError(f'Incomplete checkpoint already occupies {final_dir}')
    staging_dir = tempfile.mkdtemp(prefix=f'.{step}_', suffix='.tmp', dir=checkpoint_dir)
    try:
        with open(os.path.join(staging_dir, 'agent.pkl'), 'wb') as stream:
            pickle.dump(flax.serialization.to_state_dict(agent), stream)
        with open(os.path.join(staging_dir, 'buffer.pkl'), 'wb') as stream:
            pickle.dump(replay_buffer, stream)
        metadata = dict(step=step, online_loop_step=online_loop_step,
                        seed=seed, env_name=env_name, algo_name=algo_name,
                        wandb_run_id=wandb_run_id)
        with open(os.path.join(staging_dir, 'meta.json'), 'w') as stream:
            json.dump(metadata, stream)
        os.replace(staging_dir, final_dir)
    finally:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
    print(f'Saved complete online checkpoint: {final_dir}')
    complete_steps = sorted(int(name) for name in os.listdir(checkpoint_dir)
                            if name.isdigit() and all(os.path.isfile(
                                os.path.join(checkpoint_dir, name, filename))
                                for filename in ('agent.pkl', 'buffer.pkl', 'meta.json')))
    for old_step in complete_steps[:-2]:
        shutil.rmtree(os.path.join(checkpoint_dir, str(old_step)))


def load_rolling_checkpoint(seed: int, env_name: str, agent_structure: Any,
                            algo_name: str = 'ogpo', *, resume_path: str):
    """Restore the latest complete checkpoint only inside the requested directory.

    Both new step directories and explicitly requested legacy flat directories
    are supported. Missing or incompatible checkpoints never start a fresh run.
    """
    checkpoint_dir = os.path.abspath(os.path.expanduser(resume_path))
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f'Resume directory does not exist: {checkpoint_dir}')
    candidates = []
    for name in os.listdir(checkpoint_dir):
        if name.isdigit():
            directory = os.path.join(checkpoint_dir, name)
            paths = tuple(os.path.join(directory, filename)
                          for filename in ('agent.pkl', 'buffer.pkl', 'meta.json'))
            if all(os.path.isfile(path) for path in paths):
                candidates.append((int(name), paths))
        elif name.endswith('_agent.pkl') and name.split('_')[0].isdigit():
            step = int(name.split('_')[0])
            paths = tuple(os.path.join(checkpoint_dir, f'{step}_{suffix}')
                          for suffix in ('agent.pkl', 'buffer.pkl', 'meta.json'))
            if all(os.path.isfile(path) for path in paths):
                candidates.append((step, paths))
    for step, (agent_path, buffer_path, meta_path) in sorted(candidates, reverse=True):
        try:
            with open(meta_path) as stream:
                metadata = json.load(stream)
            with open(agent_path, 'rb') as stream:
                agent_state = pickle.load(stream)
            with open(buffer_path, 'rb') as stream:
                replay_buffer = pickle.load(stream)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
            print(f'Skipping unreadable checkpoint {step}: {error}')
            continue
        for key, expected in (('seed', seed), ('env_name', env_name), ('algo_name', algo_name)):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f'Resume {key}: {metadata[key]} != {expected}')
        online_step = metadata.get('online_loop_step', 0)
        if online_step <= 0:
            raise ValueError(f'Checkpoint {step} has no valid online step count')
        agent = flax.serialization.from_state_dict(agent_structure, agent_state)
        print(f'Resumed online checkpoint: {agent_path} (online step {online_step})')
        return agent, replay_buffer, step, online_step, metadata.get('wandb_run_id')
    raise FileNotFoundError(f'No complete readable online checkpoint in {checkpoint_dir}')


def process_train_dataset(
    ds: Dict,
    env_name: str,
    dataset_proportion: float = 1.0,
    num_offline_trajs: int = -1,
    sparse: bool = False,
    numpy_rng: Optional[np.random.Generator] = None,
    compute_mc_returns: bool = False,
    discount: float = 0.99,
) -> Dataset:
    """ Process training dataset """
    from envs.robomimic_utils import is_robomimic_env

    if numpy_rng is not None:
        ds = Dataset.create(numpy_rng=numpy_rng, **ds)
    else:
        ds = Dataset.create(**ds)

    # Trajectory-level subsampling: keep N randomly selected full rollouts.
    if num_offline_trajs > 0 and num_offline_trajs < len(ds.terminal_locs):
        rng = numpy_rng if numpy_rng is not None else np.random.default_rng(42)
        total_trajs = len(ds.terminal_locs)
        chosen = np.sort(rng.choice(total_trajs, size=num_offline_trajs, replace=False))

        keep_mask = np.zeros(ds.size, dtype=bool)
        for idx in chosen:
            s = ds.initial_locs[idx]
            e = ds.terminal_locs[idx]
            keep_mask[s:e + 1] = True

        ds_dict = {k: v[keep_mask] for k, v in ds.items()}
        ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict) if numpy_rng is not None else Dataset.create(**ds_dict)
        print(f"[process_train_dataset] Subsampled {num_offline_trajs}/{total_trajs} trajectories "
              f"({keep_mask.sum()}/{len(keep_mask)} transitions)")

    # Transition-level truncation.
    if dataset_proportion < 1.0:
        new_size = int(len(ds['masks']) * dataset_proportion)
        ds_dict = {k: v[:new_size] for k, v in ds.items()}
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)

    # Robomimic reward transformation ([0,1] -> [-1,0]) happens in
    # robomimic_utils.get_dataset(); nothing to do here.

    if sparse:
        sparse_rewards = (ds["rewards"] != 0.0) * -1.0
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards"] = sparse_rewards
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)

    if compute_mc_returns:
        mc_returns = ds.compute_mc_returns(discount)
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict['mc_returns'] = mc_returns
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)

    return ds


def get_q_bounds(env_name: str, discount: float) -> Tuple[float, float]:
    """Return (q_min, q_max) bounds for the environment."""
    from envs.robomimic_utils import is_robomimic_env
    from envs.env_utils import GYMNASIUM_ROBOTICS_ADROITHAND_ENVS

    if is_robomimic_env(env_name):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    elif env_name.startswith('PutBallintoBowl'):
        q_min = 0.0 / (1 - discount)
        q_max = 1.0 / (1 - discount)
    elif env_name.startswith('cube-triple'):
        q_min = -3.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    elif env_name.startswith('D4RL/kitchen'):
        q_min = 0 / (1 - discount)
        q_max = 4 / (1 - discount)
    elif env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        q_min = -50
        q_max = 10
    elif env_name.startswith('stacking') or env_name.startswith('gate_insertion'):
        q_min = -100
        q_max = 100
    elif env_name.startswith('pusht'):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    else:
        raise ValueError(f"Unknown environment: {env_name}, please specify min max reward")

    return q_min, q_max

def create_success_buffer_batch(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    sequence_length: int,
    discount: float,
) -> Dict:
    """ Sample a batch from the success buffer """
    success_batch = replay_buffer.sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
        discount=discount,
        success_only=True,
    )
    return success_batch

def setup_experiment_logging(
    seed: int,
    project: str,
    run_group: str,
    wandb_name: str,
    save_dir: str,
    env_name: str,
    log_enabled: bool = True,
    prefixes: Optional[List[str]] = None,
    run_timestamp: Optional[str] = None,
    resume_wandb_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[str, LoggingHelper, Optional[str]]:
    """ Setup experiment logging infrastructure """
    import wandb

    if prefixes is None:
        prefixes = ["eval", "offline_agent", "online_agent", "env"]

    run_timestamp = run_timestamp or make_run_timestamp()
    if wandb_name == "test":
        exp_name = get_exp_name(seed, run_timestamp)
    else:
        exp_name = timestamped_run_name(wandb_name + "seed" + str(seed), run_timestamp)

    # Relocate experiment data (flags.json, CSV logs) with the `OGPO_EXP_ROOT`
    # environment variable (e.g. point it at fast scratch storage on a cluster).
    # It replaces the root of `save_dir` while keeping the per-algo subdir, so
    # `exp/ogpo` -> `$OGPO_EXP_ROOT/ogpo`.
    exp_root = os.environ.get("OGPO_EXP_ROOT")
    if exp_root:
        save_dir = os.path.join(exp_root, os.path.basename(save_dir.rstrip("/")))

    wandb_run_id = None

    if log_enabled:
        run = setup_wandb(
            project=project, group=run_group, name=exp_name,
            mode=os.environ.get("WANDB_MODE", "online"), config=config,
            id=resume_wandb_id,
            resume="allow" if resume_wandb_id else None,
        )
        wandb_run_id = run.id

        save_dir = os.path.join(save_dir, run.project, run_group, env_name, exp_name)
        os.makedirs(save_dir, exist_ok=True)

        flag_dict = get_flag_dict(config)
        with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

        csv_loggers = {
            prefix: CsvLogger(os.path.join(save_dir, f"{prefix}.csv"))
            for prefix in prefixes
        }

        logger = LoggingHelper(
            csv_loggers=csv_loggers,
            wandb_logger=wandb,
            log_enabled=True,
        )
    else:
        save_dir = os.path.join(save_dir, project, run_group, env_name, exp_name)
        os.makedirs(save_dir, exist_ok=True)
        logger = LoggingHelper(
            csv_loggers={prefix: None for prefix in prefixes},
            wandb_logger=None,
            log_enabled=False,
        )

    return save_dir, logger, wandb_run_id

def get_actor_fn_for_bc_eval(agent: Any, config: Dict) -> Callable:
    """ Get the appropriate actor function for BC evaluation """
    return agent.compute_flow_actions

def get_actor_fn_for_rl_eval(agent: Any, config: Dict, use_ode: bool = False) -> Callable:
    """ Get the appropriate actor function for RL evaluation """
    if use_ode:
        return agent.compute_flow_actions
    else:
        if hasattr(agent, 'sample_actions'):
            return agent.sample_actions
        else:
            return agent.compute_flow_actions


