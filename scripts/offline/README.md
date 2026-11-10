# Shared Robomimic offline flow

Run commands from the repository root. Train once per task and seed:

```bash
bash scripts/offline/train.sh square --seed=1 --device=0
bash scripts/offline/train.sh transport --seed=1 --device=0
bash scripts/offline/train.sh toolhang --seed=1 --device=0
```

The wrapper runs `ogpo/train_offline.py` with the corresponding OGPO task
script's full configuration. It only runs BC, never resumes an online rolling
checkpoint, and exports `checkpoints/offline/<env_name>/seed_<seed>.pkl`.
Use `--export_flow_path=/absolute/path/flow.pkl` to choose another output path.

| Task | Maximum BC steps | Early stopping | Check interval | Flow |
| --- | ---: | --- | ---: | --- |
| Square | 500,000 | SDE / Best-of-8 success >= 50% | 5,000 | 4 x 512 |
| Transport | 1,000,000 | Disabled | 5,000 | 5 x 512 |
| ToolHang | 500,000 | SDE / Best-of-8 success >= 50% | 10,000 | 5 x 512 |

The existing OGPO BC runner handles sampling, optimization, evaluation, and
stopping. The portable checkpoint contains its EMA flow (used by SDE/BON),
not the critic or optimizer states. The online actor and its EMA copy both
start from these same weights when loading the checkpoint.

For example, use the same Square checkpoint in both methods:

```bash
bash scripts/dsrl/square.sh --seed=1 --flow_checkpoint_path="$PWD/checkpoints/offline/square-mh-low_dim/seed_1.pkl"
bash scripts/ogpo/square.sh --seed=1 --flow_checkpoint_path="$PWD/checkpoints/offline/square-mh-low_dim/seed_1.pkl"
```

Use `transport-mh-low_dim` and `tool_hang-ph-low_dim` for the other task paths.
Loading skips BC and always starts a fresh online experiment. Omitting both
`--flow_checkpoint_path` and `--resume_path` runs fresh BC then online training.
Old online checkpoints are never searched unless `--resume_path` is supplied.
The two options are mutually exclusive.

To enable complete online checkpoints (including replay and optimizer state):

```bash
bash scripts/ogpo/square.sh --seed=1 --flow_checkpoint_path=/path/to/flow.pkl --preemption_backup_interval=100000
```

The directory printed at startup is
`checkpoints/<algo>/<seed>_<env>/<UTC_timestamp>/`. Each complete saved step has
`agent.pkl`, `buffer.pkl`, and `meta.json` in a step subdirectory. Saving commits
that directory atomically and retains the latest two checkpoints. A final online
checkpoint is also saved when this option is enabled. DSRL saves when it crosses
the requested step threshold, even when a chunk ends between exact multiples.

Resume by passing that timestamped experiment directory (not an individual pkl):

```bash
bash scripts/ogpo/square.sh --seed=1 --resume_path=/path/to/timestamped_experiment --preemption_backup_interval=100000
```

The latest complete readable checkpoint in that directory is loaded; missing or
incompatible checkpoints fail explicitly. Compatible legacy flat checkpoint
directories can also be selected explicitly. W&B resumes the stored run ID only
for explicit online resumes; new experiments create timestamped run names.
The existing `save_interval` controls parameter-only exports, which are also
saved under timestamped log directories and are not full online resume snapshots.

Other baselines launched through `ogpo/main.py` can use
`--checkpoint.flow_path=/absolute/path/flow.pkl` with a compatible flow
architecture. Transport and ToolHang need five 512-wide flow layers, Square
four. Task, flow settings, parameter structure and shapes are checked before
loading; SAC actors and critics are not replaced.

DSRL replay now stores `chunk_steps` and `discounts`, with discounted rewards
and gamma raised to the actual executed length. Legacy DSRL replay checkpoints
are incompatible: start fresh instead of resuming the old chunk/reward schema.


## Offline flow from an inverse-trained Gaussian

Run from the repository root, with the same task hyperparameters as `train.sh`:

```bash
bash scripts/offline/train_invertion.sh square --seed=1 --device=0 \
  --flow_checkpoint_path="$PWD/checkpoints/offline/square-mh-low_dim/seed_1.pkl" \
  --base_bc_steps=100000 --training_step=90000 --early_stop=false
```

Omit `--flow_checkpoint_path` to first train a standard-normal teacher using the
original offline BC runner. The script uses DSRLResidual's existing inverse and
Gaussian encoder-BC implementations, resets the decoder and optimizer, and
retrains with `bc_update(..., use_base_distribution=True)`. The encoder stays
frozen during decoder retraining.

`--training_step` controls only decoder retraining and defaults to **90,000**.
`--early_stop` defaults to **false**, so all requested retraining updates run.
With `--early_stop=true`, the original BC runner applies the task's existing
`clip_bc_wrt` selection (ODE or SDE/BON) and success threshold. Teacher training
keeps the task's original `bc_pi_steps` and early stopping settings.

`--use_one_varience=True` (default **false**, spelling intentional) uses
`N(mu_encoder(s), I)` for decoder retraining and its ODE/SDE evaluations.
It reuses the existing `use_one_variance` implementation; inversion and encoder
training still learn the original mean and per-dimension standard deviations.
The selected source setting is recorded in W&B and the exported checkpoint config.

The same original offline runner and evaluation functions are used for both
stages: `eval/success` is ODE success using the current decoder;
`eval_sde/success` is real SDE/BON success using the EMA decoder. The SDE result is
also recorded as **`eval/success_sde`**. Both samplers start from the trained
encoder distribution during retraining. The existing shared SDE sampler accepts
an optional Gaussian source; its default N(0,I) path is unchanged. OGPO's own EMA
update is reused, with online and EMA weights reset together before retraining.

Preparation and final retraining use separate W&B runs. The final run has an
`_invertion_offline` suffix, starts at step zero, and preserves the original
`offline_agent/bc_loss`, `offline_agent/bc_flow_loss_vel`,
`offline_agent/bc_flow_loss_denoising`, `offline_agent/grad/*`, `eval/*`,
`eval_sde/*`, and `step_data/*` metric names. Encoder metrics use
`inverse_encoder/*` in the preparation run.

`--base_bc_steps` defaults to 100,000. The final export defaults to
`checkpoints/invertion_offline/<env_name>/seed_<seed>.pkl`; `--export_flow_path`
overrides it. The `ogpo_invertion_offline_v1` payload contains decoder and encoder
parameters and requires the encoder when loaded. Full agent snapshots also
contain the EMA and original SDE noise network. This entrypoint supports low_dim
state observations and standard velocity flow matching.
