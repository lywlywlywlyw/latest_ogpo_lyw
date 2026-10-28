#!/bin/bash
set -euo pipefail
# Usage: bash scripts/offline/train.sh square --seed=1 --device=0
# Optional: --export_flow_path=/path/to/flow.pkl;
# default output: checkpoints/offline/<env_name>/seed_<seed>.pkl
TASK="${1:?Usage: train.sh square|transport|toolhang [--seed=N] [--device=N]}"
shift
case "$TASK" in
  square|transport|toolhang) ;;
  *) echo "Unknown task: $TASK" >&2; exit 1 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../ogpo/$TASK.sh" "$@" --entrypoint=ogpo/train_offline.py
