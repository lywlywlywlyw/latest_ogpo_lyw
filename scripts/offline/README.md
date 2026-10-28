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
