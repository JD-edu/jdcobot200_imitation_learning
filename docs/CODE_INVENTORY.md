# JDCobot200 code inventory

This inventory records the repository state before the staged refactor. The
old code and numbered directories now live under `projects/` as historical
experiments/course material; maintained runtime code starts under
`jdcobot200/`.

| Area | Historical locations | Canonical target | Current status |
|---|---|---|---|
| Servo protocol | root `motor_drive.py` plus per-folder symlinks | `jdcobot200/hardware/motor_drive.py` | canonicalized; root module is a compatibility shim |
| Calibration/config | `projects/config/jdcobot200/offsets.txt`, per-folder `joint_limits.txt`, CLI constants | `config/jdcobot200/robot.yaml` | canonical schema and loader added |
| Hardware bring-up | `projects/1_jdcobot200_hardware_control/` | `hardware_setup/` | hardware-tested history retained |
| URDF generation | `2_jdcobot200_URDF_gen/`, `3_URDF_to_MJCF/` | `robot_description/urdf/` | canonical model selected |
| MuJoCo model | `4_*`, `5_*`, `MUJOCO_simulation/`, later pipeline folders | `robot_description/mjcf/` | canonical robot body selected; scenes separated |
| FK | `projects/6_jdcobot200_FK/` | future `jdcobot200/kinematics/` | next refactor phase |
| Safety/workspace | `projects/7_FK_safe_zone/` | future `jdcobot200/safety/` | next refactor phase |
| IK | `projects/8_jdcobot200_IK/`, `projects/MUJOCO_simulation/110_MUJOCO_IK/` | future `jdcobot200/kinematics/` | duplicated solvers identified |
| Camera calibration/pick-place | `projects/9_jdcobot200_pick_and_place/` | future `jdcobot200/perception/` and `control/` | experimental, partly hardware-tested |
| Teaching | `9_robot_teaching_data_collection/` | future `apps/record_dataset.py` | experimental UI variants retained |
| Synthetic demonstrations | `projects/10_create_virtual_dataset/` | future `jdcobot200/data/` | working simulation pipeline |
| LeRobot conversion | `projects/11_lerobot_dataset/`, `projects/12_data_recording_using_synth_data/` | future `jdcobot200/data/` | working but duplicated |
| ACT full pipelines | both `projects/13_*` directories | future `jdcobot200/policies/` and execution programs | working variants; weld/contact behavior differs |
| Astra driver | `projects/depth_cam/` | future camera adapter or separate ROS workspace | source and generated workspace currently mixed |

## Canonical decisions in phase 1

- Joint/action order is `[base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]`.
- IDs 1–5 are arm joints; ID 6 is the gripper.
- The established IK sign vector is `[1, 1, 1, -1, 1]`; gripper direction is
  provisionally `1`.
- File-based encoder offsets use `raw_home_tick - 2048`. Therefore a software
  calibrated raw tick is `logical_tick + file_offset`. The legacy driver's
  `set_offset_position()` instead consumes the servo EEPROM's native homing
  offset and retains its subtraction behavior; these two offset sources must
  not be mixed.
- Gripper offset is provisionally zero because the current shared offsets file
  contains no ID 6 entry. It must be calibrated before relying on logical zero.
- The canonical URDF SHA-256 is
  `efbdf7b783470cdbf8776ef58425e89815908d8578722a84b67ef1c61f1258de`.
- The promoted source MJCF SHA-256 was
  `92f16cd825574a2722ee748e2bfb341619b8463bd7afb01ba6ecf38d32840826`.
  Its canonical copy changes only `meshdir` to the shared mesh location and has
  SHA-256 `3635d4bf1bc439173093c94af442b5f1ee5132081df53956818432e1cc116030`.

## Generated/runtime content

Datasets, checkpoints, videos, Python caches, and ROS `build/install/log`
directories are runtime artifacts, not source. Existing artifacts are left in
place for now and are already covered by `.gitignore`; moving/removing them is
deferred so this phase does not destroy experiment results.

## Known follow-up work

1. Convert calibration applications to update `robot.yaml` atomically.
2. Replace duplicated `motor_control.py` implementations with applications
   using the canonical driver.
3. Extract FK, collision checks, and Mink IK into package modules.
4. Change historical simulation scripts to use canonical model paths.
5. Separate the Astra ROS workspace from project-owned camera adapters.
