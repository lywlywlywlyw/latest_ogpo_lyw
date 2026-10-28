"""OGPO offline BC phase, shared by online and standalone entrypoints."""

from ogpo.runners.bc_runner import run_bc_training

def run_offline_pretraining(*, agent, train_dataset, eval_env, eval_envs, config,
                            agent_config, logger, save_dir, log_step, encode_fn=None, mesh=None):
    train_config = config['training']
    eval_config = config['eval']
    interval_config = config['intervals']
    bc_config = config['bc']
    bc_pi_steps = train_config['bc_pi_steps']
    horizon_length = agent_config['horizon_length']
    discount = agent_config['discount']
    n_eval_envs = eval_config['n_eval_envs']
    run_name = config['experiment']['run_name']
    env_name = config['env']['name']
    agent, log_step, eval_info = run_bc_training(
        agent=agent,
        train_dataset=train_dataset,
        eval_env=eval_env,
        eval_envs=eval_envs,
        config=agent_config,
        logger=logger,
        save_dir=save_dir,
        bc_pi_steps=bc_pi_steps,
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
        save_interval_bc=interval_config["save_interval_bc"],
        step_data_log_interval=interval_config["step_data_log_interval"],
        clip_bc=bc_config["clip_bc"],
        clip_bc_threshold=bc_config["clip_bc_threshold"],
        clip_bc_wrt=bc_config.get("clip_bc_wrt", "ODE"),
        env_name=env_name,
        start_step=log_step,
        encode_fn=encode_fn,
        mesh=mesh,
    )
    return agent, log_step, eval_info
