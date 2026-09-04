# JDCobot200 dual-camera imitation-learning full course

This directory is a headless, reproducible pipeline from validated differential-IK demonstrations to a two-camera LeRobot v3 dataset, ACT training, and closed-loop MuJoCo inference.

## Camera setup

`jdcobot200.xml` mounts `wrist_camera` rigidly under `gripper_assembly`. `scene.xml` also defines `front_camera`. Policy inputs are:

- `observation.images.front`: RGB, 320 x 240
- `observation.images.wrist`: RGB, 320 x 240
- `observation.state`: five arm joints and one gripper joint

## 1. Generate 50 valid IK episodes and convert them

```bash
MUJOCO_GL=egl /home/robo/anaconda3/envs/lerobot/bin/python3 \
  create_lerobot_dataset.py --episodes 50 --width 320 --height 240 \
  --frame-stride 5 --overwrite
```

Only episodes passing finite-value, joint-limit, floor-clearance, non-adjacent self-collision, and placement-success checks are retained. Rejected candidates are summarized in `synthetic_dataset/manifest.json`. IK is generated and retained at 50 Hz; the default frame stride records synchronized policy observations at 10 Hz for practical headless CPU rendering.

Validate the resulting videos and decoded training samples:

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 validate_lerobot_dataset.py \
  --root lerobot_dataset --repo-id local/jdcobot200_dual_camera_pick_place \
  --expected-episodes 50 --expected-width 320 --expected-height 240
```

## 2. Train ACT

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 train_lerobot_act.py \
  --dataset-root lerobot_dataset --repo-id local/jdcobot200_dual_camera_pick_place \
  --steps 20000 --batch-size 8 --num-workers 0 \
  --chunk-size 50 --n-action-steps 50 --device auto \
  --output-dir outputs/act_jdcobot200
```

The standard checkpoint is written under `outputs/act_jdcobot200/checkpoints/last/pretrained_model`.

## 3. Run closed-loop inference headlessly

```bash
MUJOCO_GL=egl /home/robo/anaconda3/envs/lerobot/bin/python3 \
  infer_lerobot_act_mujoco.py --headless --no-realtime --steps 500 \
  --repo-id local/jdcobot200_dual_camera_pick_place
```

The rollout summary, final block pose, goal error, and success flag are saved to `outputs/inference_report.json`.

Inference uses the same grasp assist as demonstration generation. When the
tool center is within 55 mm and the policy commands a closed gripper, the block
is aligned to the demonstrated tool-center grasp pose and welded. Release requires the block to be near the goal,
low over the table, and an open command for consecutive control steps. A
completed episode cannot re-weld the block. Thresholds are exposed through the
`--grasp-*` and `--release-*` arguments.
