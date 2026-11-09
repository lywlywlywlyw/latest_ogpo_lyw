#!/bin/bash
set -euo pipefail
# Same task defaults as train.sh, with optional teacher and encoder settings.
# Example: bash scripts/offline/train_invertion.sh square --device=0 \
#   --flow_checkpoint_path=checkpoints/offline/square-mh-low_dim/seed_1.pkl \
#   --base_bc_steps=100000 --training_step=90000 --early_stop=false
TASK="${1:?Usage: train_invertion.sh square|transport|toolhang [--key=value]}"
shift
case "$TASK" in
  square|transport|toolhang) ;;
  *) echo "Unknown task: $TASK" >&2; exit 1 ;;
esac
# Handle inversion-only options here; the original OGPO task scripts do not
# know these flags. Scoped environment variables pass them to our Python entrypoint.
task_args=()
for arg in "$@"; do
  case "$arg" in
    --base_bc_steps=*|--agent.base_bc_steps=*)
      export OGPO_INVERTION_BASE_BC_STEPS="${arg#*=}" ;;
    --training_step=*)
      export OGPO_INVERTION_TRAINING_STEP="${arg#*=}" ;;
    --early_stop=*)
      export OGPO_INVERTION_EARLY_STOP="${arg#*=}" ;;
    --early_stop)
      export OGPO_INVERTION_EARLY_STOP=true ;;
    *) task_args+=("$arg") ;;
  esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../ogpo/$TASK.sh" "${task_args[@]}" --entrypoint=ogpo/train_invertion_offline.py
