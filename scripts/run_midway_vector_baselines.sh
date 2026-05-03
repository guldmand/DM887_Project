#!/usr/bin/env bash
# Run the midway vector-observation baseline matrix:
#   algorithms : sac, td3, ppo
#   envs       : cartpole_swingup, acrobot_swingup
#   seeds      : 0..4 (from --mode midway preset)
#   max_steps  : 10_000
#   eval_freq  : 1_000
#   eval_eps   : 3
#   device     : cpu
#   time cap   : 15 minutes per run
#
# The runner enforces PPO-specific overrides automatically (warmup_steps=0,
# learn_frequency=256), so a single command per env with --algorithm all is fine.
#
# Usage:
#   scripts/run_midway_vector_baselines.sh                  # real run (requires --allow-batch-run, set below)
#   DRY_RUN=1 scripts/run_midway_vector_baselines.sh        # dry-run preview only
#
# Run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-conda run -n RL python}"
RUN_FLAG="--run"
EXTRA_FLAGS="--allow-batch-run"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUN_FLAG="--dry-run"
  EXTRA_FLAGS=""
fi

ENVS=(cartpole_swingup acrobot_swingup)

for env in "${ENVS[@]}"; do
  echo "=============================="
  echo "ENV: ${env}"
  echo "=============================="
  ${PYTHON} scripts/run_project_objectrl_baseline.py \
    --mode midway \
    --algorithm all \
    --project-env "${env}" \
    --max-steps 10000 \
    --warmup-steps 100 \
    --eval-frequency 1000 \
    --eval-episodes 3 \
    --device cpu \
    --time-limit-minutes 15 \
    ${EXTRA_FLAGS} \
    ${RUN_FLAG}
done

echo
echo "Done. Summarize with:"
echo "  ${PYTHON} scripts/summarize_project_baselines.py --prefix midway"
