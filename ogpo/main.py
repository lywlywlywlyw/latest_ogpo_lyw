import glob
import os
import random

import jax
import numpy as np

from ogpo.utils.log_utils import make_run_timestamp, is_run_timestamp
from ogpo.utils.sharding import create_mesh, shard_agent

if "CUDA_VISIBLE_DEVICES" in os.environ:
    # Use the first visible GPU for MuJoCo EGL rendering
    _egl_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    os.environ["EGL_DEVICE_ID"] = _egl_device
    os.environ["MUJOCO_EGL_DEVICE_ID"] = _egl_device

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from ogpo.utils.config_loader import load_config, parse_cli_args, config_to_agent_dict
from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.main_helper import (
    make_vectorized_env,
    get_checkpoint_dir,
    load_rolling_checkpoint,
    process_train_dataset,
    get_q_bounds,
    setup_experiment_logging,
)
from ogpo.runners.bc_runner import (
    run_bc_q_training,
    run_calql_training,
    _build_eval_flags,
    _run_evaluation,
)
from ogpo.runners.offline_runner import run_offline_pretraining
from ogpo.utils.flow_checkpoint import load_flow_checkpoint, save_flow_checkpoint
from ogpo.runners.online_rl_runner import run_online_rl
from ogpo.runners.dsrl_runner import run_dsrl_online_rl, DSRL_ALGORITHMS
from ogpo.agents.ogpo import OGPOAgent
from ogpo.agents.qc import ACFQLAgent
from ogpo.agents.bptt import OGPOBPTTAgent
from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.dsrlna import DSRLNAAgent
from ogpo.agents.jgcw import JGCWAgent
from ogpo.runners.jgcw_runner import run_jgcw_online_rl
from ogpo.agents.expo import EXPOAgent
from ogpo.agents.dsrl_plus_expo import DSRLEXPOAgent


def _load_frozen_encoder(encoder_path, agent_config):
    """Load a frozen encoder from pkl and return a JIT-compiled encode function."""
    import pickle
    from ogpo.networks.encoders import VisionProprioEncoder
    from ogpo.networks.encoders import vit_encoder_modules as _vit_modules

    with open(encoder_path, 'rb') as f:
        encoder_params = pickle.load(f)

    vit = _vit_modules[agent_config['vit_encoder']]()
    encoder = VisionProprioEncoder(
        vit=vit,
        state_proj_dim=agent_config['vit_state_proj_dim'],
        img_proj_dim=agent_config.get('vit_img_proj_dim', 0),
    )
    frozen_params = {'params': encoder_params}

    @jax.jit
    def encode_fn(observations, images):
        return encoder.apply(frozen_params, images, observations)

    return encode_fn


def _pre_encode_dataset(dataset, encode_fn, batch_size=512, agent_config=None):
    """Replace (observations, images) with encoded embeddings in-place.

    `agent_config['use_state']` selects the state vector concatenated with image
    features: 'full' uses full_states (proprio + object pose), 'proprio' uses the
    robot proprio only. When provided, stashes the two-tier img/proprio dims so
    the obs encoder can auto-size its proprio tower.
    """
    n = dataset.size
    imgs = dataset._dict['images']
    next_imgs = dataset._dict['next_images']

    use_state = (agent_config or {}).get('use_state', 'proprio')
    if use_state not in ('full', 'proprio'):
        raise ValueError(f"use_state must be 'full' or 'proprio', got {use_state!r}")

    has_full = 'full_states' in dataset._dict and 'next_full_states' in dataset._dict
    if has_full and use_state == 'full':
        obs = dataset._dict['full_states']
        next_obs = dataset._dict['next_full_states']
    else:
        obs = dataset._dict['observations']
        next_obs = dataset._dict['next_observations']
    print(f"  Pre-encode state source: use_state={use_state} -> "
          f"{'full_states' if (has_full and use_state == 'full') else 'observations'} "
          f"(proprio_dim={obs.shape[-1]})")

    proprio_dim = int(obs.shape[-1])

    sample_encoded = np.asarray(encode_fn(obs[:1], imgs[:1]))
    encoded_dim = sample_encoded.shape[-1]
    img_feat_dim = encoded_dim - proprio_dim  # encoder concats [img_features, proprio]

    if agent_config is not None:
        agent_config['_two_tier_img_dim'] = img_feat_dim
        agent_config['_two_tier_proprio_dim'] = proprio_dim
        print(f"  Two-tier split recorded: img_feat_dim={img_feat_dim}, proprio_dim={proprio_dim}")

    new_obs = np.empty((len(obs), encoded_dim), dtype=np.float32)
    new_next = np.empty((len(next_obs), encoded_dim), dtype=np.float32)

    for i in range(0, n, batch_size):
        e = min(i + batch_size, n)
        new_obs[i:e] = np.asarray(encode_fn(obs[i:e], imgs[i:e]))
        new_next[i:e] = np.asarray(encode_fn(next_obs[i:e], next_imgs[i:e]))

    dataset._dict['observations'] = new_obs
    dataset._dict['next_observations'] = new_next
    del dataset._dict['images']
    del dataset._dict['next_images']
    # full_states are now folded into the encoded observations
    if has_full:
        del dataset._dict['full_states']
        del dataset._dict['next_full_states']

AGENTS = {
    "ogpo": OGPOAgent,
    "qc": ACFQLAgent,
    "acfql": ACFQLAgent,
    "bptt": OGPOBPTTAgent,
    "ogpo_bptt": OGPOBPTTAgent,
    "jgcw": JGCWAgent,
    "dsrlna": DSRLNAAgent,
    "dsrl": DSRLBestOfNAgent,
    "dsrl_bon": DSRLBestOfNAgent,
    "expo": EXPOAgent,
    "dsrl_plus_expo": DSRLEXPOAgent,
}


def main(*, offline_only=False):
    algo_name, cli_overrides = parse_cli_args()
    config = load_config(algo_name, cli_overrides)

    if "agent.agent_name" in cli_overrides:
        algo_name = cli_overrides["agent.agent_name"]
    else:
        algo_name = config["agent"]["agent_name"]
    print(f"Running algorithm: {algo_name}")
    flow_path = config['checkpoint'].get('flow_path')
    resume_path = config['checkpoint'].get('resume_path')
    if flow_path and resume_path:
        raise ValueError('flow_checkpoint_path and resume_path are mutually exclusive')
    if resume_path and (config['checkpoint'].get('restore_actor_path') or
                        config['checkpoint'].get('restore_critic_path')):
        raise ValueError('resume_path cannot be combined with individual weight restores')
    if flow_path and config['checkpoint'].get('restore_actor_path'):
        raise ValueError('Use either flow_checkpoint_path or restore_actor_path')
    if offline_only and resume_path:
        raise ValueError('Offline pretraining does not accept an online resume_path')
    if resume_path and not os.path.isdir(os.path.expanduser(resume_path)):
        raise FileNotFoundError(f'Resume directory does not exist: {resume_path}')

    if offline_only:
        if algo_name != "ogpo":
            raise ValueError("Standalone offline pretraining requires --algo=ogpo")
        if config['training']['bc_pi_steps'] <= 0:
            raise ValueError("Offline pretraining requires training.bc_pi_steps > 0")
        config['training'].update(bc_only=True, online_steps=0, bc_q_steps=0, calql_steps=0)
        if config['checkpoint'].get('flow_path'):
            raise ValueError("Offline pretraining expects no checkpoint.flow_path")
        if not config['checkpoint'].get('export_flow_path'):
            config['checkpoint']['export_flow_path'] = os.path.join(
                'checkpoints', 'offline', config['env']['name'],
                f"seed_{config['experiment']['seed']}.pkl")

    exp_config = config["experiment"]
    env_config = config["env"]
    train_config = config["training"]
    buffer_config = config["buffer"]
    eval_config = config["eval"]
    interval_config = config["intervals"]
    bc_config = config["bc"]
    success_buffer_config = config["success_buffer"]
    dataset_config = config["dataset"]
    checkpoint_config = config["checkpoint"]
    agent_config = config_to_agent_dict(config)

    if algo_name not in AGENTS:
        raise ValueError(
            f"Unknown agent: {algo_name}. Available: {list(AGENTS.keys())}"
        )
    agent_class = AGENTS[algo_name]

    env_name = env_config["name"]
    seed = exp_config["seed"]
    ogbench_dataset_dir = dataset_config["ogbench_dir"]

    if ogbench_dataset_dir is not None:
        assert dataset_config["replace_interval"] != 0
        assert dataset_config["proportion"] == 1.0
        dataset_paths = [
            f
            for f in sorted(glob.glob(f"{ogbench_dataset_dir}/*.npz"))
            if "-val.npz" not in f
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            env_name, dataset_path=dataset_paths[0], compact_dataset=False
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            env_name, rew_fn=env_config["rew_fn"],
            post_success_steps=env_config.get("post_success_steps", 0),
        )

    del val_dataset  # free immediately to save RAM

    n_eval_envs = eval_config["n_eval_envs"]
    eval_envs = None
    if n_eval_envs > 1:
        eval_envs = make_vectorized_env(
            env_name,
            num_envs=n_eval_envs,
            seed=seed + 1000,
            rew_fn=env_config["rew_fn"],
            post_success_steps=env_config.get("post_success_steps", 0),
        )

    random.seed(seed)
    np.random.seed(seed)
    dataset_rng = np.random.default_rng(seed + 20000)
    online_rng, rng = jax.random.split(jax.random.PRNGKey(seed), 2)

    horizon_length = agent_config["horizon_length"]
    discount = agent_config["discount"]
    best_of_n = agent_config["best_of_n"]

    q_min, q_max = get_q_bounds(env_name, discount)
    agent_config["q_min"] = q_min
    agent_config["q_max"] = q_max
    print(f"Q bounds: [{q_min}, {q_max}]")

    needs_mc_returns = algo_name in [
        "ogpo",
        "reinflow_calql",
        "bptt",
        "ogpo_bptt",
        "expo",
    ]
    train_dataset = process_train_dataset(
        train_dataset,
        env_name,
        dataset_proportion=dataset_config["proportion"],
        num_offline_trajs=dataset_config.get("num_offline_trajs", -1),
        sparse=env_config["sparse"],
        numpy_rng=dataset_rng,
        compute_mc_returns=needs_mc_returns,
        discount=discount,
    )
    train_dataset.p_aug = dataset_config["p_aug"]

    # Save original actor/critic obs before any frozen-encoder overrides (mixed obs mode)
    agent_config['_original_actor_obs'] = agent_config.get('actor_obs', 'state')
    agent_config['_original_critic_obs'] = agent_config.get('critic_obs', 'state')

    encode_fn = None
    restore_encoder_path = checkpoint_config.get("restore_encoder_path")
    is_paligemma = agent_config.get('vit_encoder', '') == 'paligemma'

    if is_paligemma and agent_config.get('freeze_vision_encoder', False):
        # PaliGemma frozen VL encoder: SigLIP vision tower + Gemma LM trunk from
        # a Pi0/Pi0.5 checkpoint. The LM trunk runs per-task instructions.
        from ogpo.networks.encoders.paligemma import load_paligemma_encoder
        paligemma_ckpt = restore_encoder_path if (restore_encoder_path and restore_encoder_path != "None") \
            else os.path.expanduser("~/checkpoints/pi05_base/params")
        encoder_device = agent_config.get('encoder_device', None)
        if encoder_device is not None:
            encoder_device = int(encoder_device)
        tokenizer_path = agent_config.get('paligemma_tokenizer_path') or None
        print(f"Loading PaliGemma frozen VL encoder: {paligemma_ckpt} (encoder GPU: {encoder_device})")
        encode_fn = load_paligemma_encoder(
            checkpoint_path=paligemma_ckpt,
            env_name=env_name,
            img_proj_dim=agent_config.get('vit_img_proj_dim', 0),
            state_proj_dim=agent_config.get('vit_state_proj_dim', 0),
            encoder_device=encoder_device,
            tokenizer_path=tokenizer_path,
        )

        if 'images' in train_dataset._dict:
            # Small batch for PaliGemma (SigLIP is large; 512 causes OOM)
            _pre_encode_dataset(train_dataset, encode_fn, batch_size=32, agent_config=agent_config)
            print(f"  Pre-encoded {train_dataset.size} observations. "
                  f"New obs dim: {train_dataset['observations'].shape[-1]}")

        agent_config['actor_obs'] = 'state'
        agent_config['critic_obs'] = 'state'
        agent_config['_encoder_frozen'] = True
        train_dataset.p_aug = None

    elif (agent_config.get('freeze_vision_encoder', False)
            and restore_encoder_path and restore_encoder_path != "None"):
        print(f"Loading frozen encoder from {restore_encoder_path}")
        encode_fn = _load_frozen_encoder(restore_encoder_path, agent_config)

        if 'images' in train_dataset._dict:
            _pre_encode_dataset(train_dataset, encode_fn, agent_config=agent_config)
            print(f"  Pre-encoded {train_dataset.size} observations. "
                  f"New obs dim: {train_dataset['observations'].shape[-1]}")

        # Override config so agent is created as pure MLP (no encoder submodule)
        agent_config['actor_obs'] = 'state'
        agent_config['critic_obs'] = 'state'
        agent_config['_encoder_frozen'] = True
        train_dataset.p_aug = None  # no image augmentation on embeddings

    example_batch = train_dataset.sample(())
    action_dim = example_batch["actions"].shape[-1]
    ex_images = example_batch.get("images")
    ex_full_states = example_batch.get("full_states")

    if algo_name in ["ogpo", "reinflow_calql"]:
        agent = agent_class.create(
            seed,
            example_batch["observations"],
            example_batch["actions"],
            agent_config,
            agent_config["adv_clip_min"],
            ex_images=ex_images,
            ex_full_states=ex_full_states,
        )
    else:
        agent = agent_class.create(
            seed, example_batch["observations"], example_batch["actions"], agent_config
        )

    gpu_mode = train_config.get("gpu_mode", "single")
    mesh = create_mesh(gpu_mode)
    if mesh is not None:
        num_mesh_devices = len(mesh.devices)
        if agent_config['batch_size'] % num_mesh_devices != 0:
            raise ValueError(
                f"batch_size ({agent_config['batch_size']}) must be divisible by "
                f"mesh devices ({num_mesh_devices})")
        agent = shard_agent(agent, mesh)
        print(f"Multi-GPU ({gpu_mode}): {num_mesh_devices} devices, "
              f"{agent_config['batch_size'] // num_mesh_devices} samples/device")

    prefixes = ["eval", "env", "offline_agent", "memory"]
    online_steps = train_config["online_steps"]
    bc_only = train_config["bc_only"]
    if online_steps > 0 and not bc_only:
        prefixes.append("online_agent")
        if algo_name == "jgcw":
            prefixes.append("online_budget")
    if algo_name in ["ogpo", "reinflow_calql"]:
        prefixes.extend(["eval_sde", "step_data"])
    if agent_config.get('use_one_step_policy', False):
        prefixes.append("eval_one_step")
    if train_config.get("q_warmup_steps", 0) > 0 and not bc_only:
        prefixes.append("q_warmup")

    is_resumed = False
    resumed_step = 0
    resumed_online_step = 0
    resumed_buffer = None
    resumed_wandb_id = None
    run_timestamp = make_run_timestamp()
    checkpoint_dir = get_checkpoint_dir(seed, env_name, algo_name, run_timestamp)
    if resume_path:
        (agent, resumed_buffer, resumed_step, resumed_online_step,
         resumed_wandb_id) = load_rolling_checkpoint(
            seed, env_name, agent, algo_name, resume_path=resume_path)
        if resumed_online_step >= online_steps:
            raise ValueError(f'Resume step {resumed_online_step} must be below online_steps={online_steps}')
        if bc_only:
            raise ValueError('Online resume_path cannot be used with training.bc_only=true')
        if mesh is not None:
            agent = shard_agent(agent, mesh)
        is_resumed = True
        # Continue within the original timestamped experiment. Legacy directories
        # remain readable, but subsequent saves go to a new timestamped directory.
        resumed_dir = os.path.abspath(os.path.expanduser(resume_path))
        previous_timestamp = os.path.basename(resumed_dir)
        if is_run_timestamp(previous_timestamp):
            run_timestamp = previous_timestamp
            checkpoint_dir = resumed_dir
        print(f'Startup: resume explicit online experiment at step {resumed_online_step}')
    elif flow_path:
        print(f'Startup: shared offline flow {flow_path}; new online experiment starts at step 0')
    else:
        print('Startup: fresh offline BC followed by a new online experiment')
    print(f'Online checkpoint directory (when enabled): {checkpoint_dir}')
    print(f'Resume with: --resume_path={checkpoint_dir}')

    run_name = exp_config["run_name"]
    save_dir, logger, wandb_run_id = setup_experiment_logging(
        seed=seed, project=exp_config["project"], run_group=exp_config["run_group"],
        wandb_name=exp_config["wandb_name"], save_dir=exp_config["save_dir"],
        env_name=env_name, log_enabled=exp_config["log"], prefixes=prefixes,
        run_timestamp=run_timestamp, resume_wandb_id=resumed_wandb_id, config=config,
    )

    # Restore pre-trained weights
    if not is_resumed:
        restore_actor_path = checkpoint_config["restore_actor_path"]
        restore_critic_path = checkpoint_config["restore_critic_path"]
        if restore_actor_path and restore_actor_path != "None":
            agent.restore_actor_params(restore_actor_path)
            print(f"Restored actor from {restore_actor_path}")
        if restore_critic_path and restore_critic_path != "None":
            agent.restore_critic_params(restore_critic_path)
            print(f"Restored critic from {restore_critic_path}")

    ep_resume = checkpoint_config["ep_resume"]
    log_step = ep_resume if not is_resumed else resumed_step
    bc_pi_steps = train_config["bc_pi_steps"]

    flow_path = checkpoint_config.get('flow_path')
    if not is_resumed and flow_path:
        agent = load_flow_checkpoint(agent, flow_path, env_name=env_name)
        if mesh is not None:
            agent = shard_agent(agent, mesh)
        bc_pi_steps = 0
        print(f"Loaded shared offline flow from {flow_path}; skipping BC training")

    if not is_resumed and bc_pi_steps > 0:
        print("\n" + "=" * 60 + "\nBC Training Phase\n" + "=" * 60)
        agent, log_step, eval_info = run_offline_pretraining(
            agent=agent, train_dataset=train_dataset, eval_env=eval_env,
            eval_envs=eval_envs, config=config, agent_config=agent_config,
            logger=logger, save_dir=save_dir, log_step=log_step,
            encode_fn=encode_fn, mesh=mesh,
        )
        print(
            f"BC Training Complete. Final success rate: {eval_info.get('success', 'N/A')}"
        )

    if checkpoint_config.get('export_flow_path'):
        if is_resumed:
            raise ValueError("Cannot export offline flow from a resumed online run")
        save_flow_checkpoint(agent, checkpoint_config['export_flow_path'],
                             env_name=env_name, seed=seed, step=log_step)

    bc_q_steps = train_config["bc_q_steps"]
    if not is_resumed and bc_q_steps > 0 and algo_name in ["ogpo", "reinflow_calql"]:
        print("\n" + "=" * 60 + "\nBC-Q Training Phase\n" + "=" * 60)
        agent, log_step, eval_info = run_bc_q_training(
            agent=agent,
            train_dataset=train_dataset,
            eval_env=eval_env,
            eval_envs=eval_envs,
            config=agent_config,
            logger=logger,
            save_dir=save_dir,
            bc_q_steps=bc_q_steps,
            horizon_length=horizon_length,
            discount=discount,
            batch_size=agent_config["batch_size"],
            n_eval_envs=n_eval_envs,
            eval_episodes=eval_config["eval_episodes"],
            eval_interval_bc=interval_config["eval_interval_bc"],
            video_episodes=eval_config["video_episodes"],
            video_frame_skip=eval_config["video_frame_skip"],
            plot_q_vs_mc=eval_config["plot_q_vs_mc"],
            run_name=run_name,
            log_interval=interval_config["log_interval"],
            step_data_log_interval=interval_config["step_data_log_interval"],
            env_name=env_name,
            start_step=log_step,
            mesh=mesh,
        )

    calql_steps = train_config["calql_steps"]
    if not is_resumed and calql_steps > 0 and algo_name in ["ogpo", "reinflow_calql"]:
        print("\n" + "=" * 60 + "\nCalQL Training Phase\n" + "=" * 60)
        agent, log_step, eval_info = run_calql_training(
            agent=agent,
            train_dataset=train_dataset,
            eval_env=eval_env,
            eval_envs=eval_envs,
            config=agent_config,
            logger=logger,
            save_dir=save_dir,
            calql_steps=calql_steps,
            horizon_length=horizon_length,
            discount=discount,
            batch_size=agent_config["batch_size"],
            n_eval_envs=n_eval_envs,
            eval_episodes=eval_config["eval_episodes"],
            eval_interval_bc=interval_config["eval_interval_bc"],
            video_episodes=eval_config["video_episodes"],
            video_frame_skip=eval_config["video_frame_skip"],
            plot_q_vs_mc=eval_config["plot_q_vs_mc"],
            run_name=run_name,
            log_interval=interval_config["log_interval"],
            step_data_log_interval=interval_config["step_data_log_interval"],
            env_name=env_name,
            start_step=log_step,
            encode_fn=encode_fn,
            mesh=mesh,
        )

    # Final BC evaluation (SDE) for OGPO
    if (algo_name == "ogpo") and (not bc_only):
        print("\nEvaluating with Stochastic Flow (SDE)...")
        eval_flags = _build_eval_flags(
            n_eval_envs, eval_config["eval_episodes"], discount, run_name, env_name
        )
        # Feed the encoder the same proprio width the dataset was pre-encoded with,
        # else apply raises ScopeParamShapeError.
        _actor_uses_state = agent_config.get('_original_actor_obs', 'state') == 'state'
        _use_full_state = agent_config.get('use_state', 'proprio') == 'full'
        sde_actor_fn = None
        if encode_fn is not None:
            from ogpo.runners.online_rl_runner import _wrap_actor_for_frozen_encoder
            sde_actor_fn = _wrap_actor_for_frozen_encoder(
                agent.sample_actions, encode_fn,
                actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
        eval_info = _run_evaluation(
            agent,
            eval_env,
            eval_envs,
            eval_flags,
            action_dim,
            sde_actor_fn,
            log_step,
            eval_config["eval_episodes"],
            eval_config["video_episodes"],
            eval_config["video_frame_skip"],
            run_name,
            plot=False,
        )
        print(f"SDE Success Rate: {eval_info.get('success', 'N/A')}")
        try:
            logger.log(eval_info, "eval_sde", step=log_step)
        except Exception:
            pass

        if agent_config.get('use_one_step_policy', False):
            one_step_actor_fn = agent.sample_actions_one_step_BON if best_of_n > 1 else agent.sample_actions_one_step
            # Frozen-encoder runs: wrap so env-raw obs gets re-encoded to the dim
            # the one-step MLP was init'd with.
            if encode_fn is not None:
                from ogpo.runners.online_rl_runner import _wrap_actor_for_frozen_encoder
                one_step_actor_fn = _wrap_actor_for_frozen_encoder(
                    one_step_actor_fn, encode_fn,
                    actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
            bon_str = f" (BON N={best_of_n})" if best_of_n > 1 else ""
            print(f"\nEvaluating with One-Step Policy{bon_str}...")
            eval_info_one_step = _run_evaluation(
                agent,
                eval_env,
                eval_envs,
                eval_flags,
                action_dim,
                one_step_actor_fn,
                log_step,
                eval_config["eval_episodes"],
                eval_config["video_episodes"],
                eval_config["video_frame_skip"],
                run_name,
                plot=False,
            )
            print(f"One-Step Success Rate: {eval_info_one_step.get('success', 'N/A')}")
            try:
                logger.log(eval_info_one_step, "eval_one_step", step=log_step)
            except Exception:
                pass

    if bc_only or online_steps <= 0:
        print("\nBC training complete. Skipping online RL phase.")
        save_agent(agent, save_dir, log_step)
        logger.close()
        return
    else:
        print("\nBC training complete. Starting online RL phase.")
        save_agent(agent, save_dir, log_step)

    ft_flow_steps = agent_config.get("ft_flow_steps", 0)
    if 0 < ft_flow_steps < agent_config["flow_steps"]:
        print(f"RL finetuning: IS ratio computed over last {ft_flow_steps}/{agent_config['flow_steps']} flow steps")

    print("\n" + "=" * 60 + "\nOnline RL Phase\n" + "=" * 60)

    # Late freeze: freeze after BC training, extracting the encoder from the
    # trained agent (early freeze already set encode_fn from restore_encoder_path).
    if encode_fn is None and agent_config.get('freeze_vision_encoder', False) and agent_config.get('encoder'):
        print("Freezing vision encoder and pre-encoding observations...")
        encode_fn = agent.create_frozen_encode_fn()

        if train_dataset is not None and 'images' in train_dataset._dict:
            _pre_encode_dataset(train_dataset, encode_fn)
            print(f"  Pre-encoded {train_dataset.size} observations. "
                  f"New obs dim: {train_dataset['observations'].shape[-1]}")

        agent.config['_encoder_frozen'] = True

    runner_kwargs = dict(
        agent=agent,
        env=env,
        eval_env=eval_env,
        eval_envs=eval_envs,
        train_dataset=train_dataset,
        config=agent_config,
        logger=logger,
        save_dir=save_dir,
        algo_name=algo_name,
        online_steps=online_steps,
        start_training=buffer_config["start_training"],
        batch_size=agent_config["batch_size"],
        horizon_length=horizon_length,
        discount=discount,
        utd_warmup=buffer_config["utd_warmup"],
        utd_online=buffer_config["utd_online"],
        utd_q=buffer_config.get("utd_q"),
        utd_pi=buffer_config.get("utd_pi"),
        utd_v=buffer_config.get("utd_v"),
        offline_ratio=buffer_config["offline_ratio"],
        bc_refine_steps=train_config["bc_refine_steps"],
        q_warmup_steps=train_config["q_warmup_steps"],
        use_success_buffer=success_buffer_config["use_success_buffer"],
        success_buffer_batch_size=success_buffer_config["batch_size"],
        best_of_n=best_of_n,
        eval_interval=interval_config["eval_interval"],
        n_eval_envs=n_eval_envs,
        eval_episodes=eval_config["eval_episodes"],
        video_episodes=eval_config["video_episodes"],
        video_frame_skip=eval_config["video_frame_skip"],
        plot_q_vs_mc=eval_config["plot_q_vs_mc"],
        run_name=run_name,
        env_name=env_name,
        log_interval=interval_config["log_interval"],
        step_data_log_interval=interval_config["step_data_log_interval"],
        save_interval=interval_config["save_interval"],
        preemption_backup_interval=interval_config["preemption_backup_interval"],
        checkpoint_dir=checkpoint_dir,
        log_enabled=exp_config["log"],
        sparse=env_config["sparse"],
        p_aug=dataset_config["p_aug"],
        buffer_size=buffer_config["size"],
        seed=seed,
        is_resumed=is_resumed,
        resumed_online_step=resumed_online_step,
        resumed_replay_buffer=resumed_buffer if is_resumed else None,
        start_step=log_step,
        wandb_run_id=wandb_run_id,
        force_jax_sync=train_config.get("force_jax_sync", True),
        encode_fn=encode_fn,
        mc_regression=agent_config.get("mc_regression", False),
        mesh=mesh,
    )

    if algo_name == "jgcw":
        agent, log_step, final_eval_info, replay_buffer = run_jgcw_online_rl(**runner_kwargs)
    elif algo_name in DSRL_ALGORITHMS:
        agent, log_step, final_eval_info, replay_buffer = run_dsrl_online_rl(
            **runner_kwargs
        )
    else:
        agent, log_step, final_eval_info, replay_buffer = run_online_rl(**runner_kwargs)

    print(
        f"\nOnline RL complete. Final success rate: {final_eval_info.get('success', 'N/A')}"
    )
    logger.close()


if __name__ == '__main__':
    main()
