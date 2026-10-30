#!/bin/bash

set -e

export MUJOCO_GL=egl

# === Config ===
seed=1
device=0
jax_frac=-1
env_name="transport-mh-low_dim"
run_name="jgcw_transport"
wandb_name="jgcw_transport"
project="OGPO"
run_group="transport"

# Training phases
bc_pi_steps=1000000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=6000000
gpu_mode=single

# Agent config
horizon_length=8
discount=0.999
tau=0.005
best_of_n=1
num_qs=10
q_agg="mean"
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal       # scalar | sinusoidal
time_embedding_dim=32
noise_chunk_length=-1
bc_flow_hidden_dims="[512,512,512,512,512]"

# Stability hyperparameters
action_magnitude=1.5
clip_grad_norm=1.0

# UTD and buffer
utd_warmup=1
utd_online=20
offline_ratio=0.0
start_training=40000

# Evaluation
n_eval_envs=32
eval_episodes=2
log_interval=5000
eval_interval=20000
eval_interval_bc=5000
save_interval=500000

# BC settings
clip_bc=false

# Success buffer
use_success_buffer=false

# Paths
SAVE_DIR="exp/ogpo"
restore_actor_path="null"
flow_checkpoint_path="null"
resume_path="null"
preemption_backup_interval=-1
restore_critic_path="null"
ep_resume=0

# Logging
log=true
plot_q_vs_mc=false

# === Parse CLI overrides ===
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"
      eval "$key=\"$value\""
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# Preserve the actor budget; default to twice as many critic updates.
utd_pi=${utd_pi:-$utd_online}
utd_q=${utd_q:-$((2 * utd_pi))}

export CUDA_VISIBLE_DEVICES=$device
if [[ "$jax_frac" == "-1" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset XLA_PYTHON_CLIENT_MEM_FRACTION
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac
fi

# === Run ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Project Python not found: $PYTHON_BIN. Set up .venv before training." >&2
    exit 1
fi

echo "=== JGCW Transport | seed=$seed ==="

"$PYTHON_BIN" ogpo/main.py \
    --algo=jgcw \
    --experiment.project=$project \
    --experiment.run_group=$run_group \
    --experiment.run_name=$run_name \
    --experiment.wandb_name=$wandb_name \
    --experiment.seed=$seed \
    --experiment.log=$log \
    --experiment.save_dir=$SAVE_DIR \
    --env.name=$env_name \
    --env.sparse=false \
    --training.bc_pi_steps=$bc_pi_steps \
    --training.calql_steps=$calql_steps \
    --training.q_warmup_steps=$q_warmup_steps \
    --training.bc_refine_steps=$bc_refine_steps \
    --training.online_steps=$online_steps \
    --training.bc_only=false \
    --training.gpu_mode=$gpu_mode \
    --buffer.start_training=$start_training \
    --buffer.utd_warmup=$utd_warmup \
    --buffer.utd_online=$utd_online \
    --buffer.utd_pi=$utd_pi \
    --buffer.utd_q=$utd_q \
    --buffer.offline_ratio=$offline_ratio \
    --eval.n_eval_envs=$n_eval_envs \
    --eval.eval_episodes=$eval_episodes \
    --eval.plot_q_vs_mc=$plot_q_vs_mc \
    --intervals.log_interval=$log_interval \
    --intervals.eval_interval=$eval_interval \
    --intervals.eval_interval_bc=$eval_interval_bc \
    --intervals.save_interval=$save_interval \
    --bc.clip_bc=$clip_bc \
    --success_buffer.use_success_buffer=$use_success_buffer \
    --checkpoint.restore_actor_path=$restore_actor_path \
    --checkpoint.flow_path=$flow_checkpoint_path \
    --checkpoint.resume_path=$resume_path \
    --intervals.preemption_backup_interval=$preemption_backup_interval \
    --checkpoint.restore_critic_path=$restore_critic_path \
    --checkpoint.ep_resume=$ep_resume \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.q_agg=$q_agg \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.time_embedding=$time_embedding \
    --agent.time_embedding_dim=$time_embedding_dim \
    --agent.noise_chunk_length=$noise_chunk_length \
    --agent.bc_flow_hidden_dims=$bc_flow_hidden_dims \
    --agent.action_magnitude=$action_magnitude \
    --agent.clip_grad_norm=$clip_grad_norm
