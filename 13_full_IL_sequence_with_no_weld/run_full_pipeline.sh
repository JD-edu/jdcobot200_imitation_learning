#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/robo/anaconda3/envs/lerobot/bin/python3}"
EPISODES="${1:-50}"
TRAIN_STEPS="${2:-20000}"
REPO_ID="local/jdcobot200_contact_pick_place"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" create_lerobot_dataset.py \
  --episodes "$EPISODES" --width 320 --height 240 --frame-stride 5 --overwrite

"$PYTHON_BIN" validate_lerobot_dataset.py \
  --root lerobot_dataset --repo-id "$REPO_ID" \
  --expected-episodes "$EPISODES" --expected-width 320 --expected-height 240

"$PYTHON_BIN" train_lerobot_act.py \
  --dataset-root lerobot_dataset --repo-id "$REPO_ID" \
  --steps "$TRAIN_STEPS" --batch-size 8 --num-workers 0 \
  --chunk-size 50 --n-action-steps 50 --device auto \
  --output-dir outputs/act_jdcobot200 --overwrite

MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" infer_lerobot_act_mujoco.py \
  --headless --no-realtime --steps 500 --repo-id "$REPO_ID"
