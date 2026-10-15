#!/usr/bin/env python3
"""09 과정에서 사용하는 합성 expert episode 기록 도구."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

# DISPLAY가 없는 데이터 생성 환경에서도 offscreen RGB를 렌더링한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


LESSON_DIR = Path(__file__).resolve().parent
CURRICULUM_DIR = LESSON_DIR.parent
EXPERT_PATH = CURRICULUM_DIR / "08_synthetic_pick_and_place/02_randomized_pick_and_place.py"
SCENE_PATH = CURRICULUM_DIR / "08_synthetic_pick_and_place/scenes/fixed_pick_and_place.xml"
SCHEMA_VERSION = 2
# 비대칭 finger pad가 닫히며 블록을 약 40도 회전시키는 물리 효과를 상쇄한다.
# 목표 marker의 yaw는 바꾸지 않고 place 구간의 jaw 명령만 보정한다.
PHYSICS_PLACE_YAW_COMPENSATION_RAD = np.deg2rad(40.0)
PHYSICS_PLACE_POSITION_COMPENSATION_LOCAL_M = np.array([0.0200, -0.0337])
FEATURES = {
    "observation.images.dataset_camera": "uint8[T,H,W,3] RGB",
    "observation.state": "float32[T,6] arm radians + gripper radians",
    "action": "float32[T,6] commanded arm + gripper radians",
    "timestamp": "float64[T] seconds",
    "phase": "uint8[T] phase index",
    "tcp_position": "float32[T,3] privileged debug value in metres",
    "block_pose": "float32[T,4] privileged debug x,y,z,yaw",
}


def _load_expert():
    spec = importlib.util.spec_from_file_location("curriculum08_expert", EXPERT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"08 expert를 읽을 수 없습니다: {EXPERT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPERT = _load_expert()


def episode_path(dataset_dir: Path, episode_index: int) -> Path:
    return dataset_dir / "episodes" / f"episode_{episode_index:06d}.npz"


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sample_plan(
    rng: np.random.Generator,
    *,
    tolerance_mm: float,
    path_steps: int,
    max_attempts: int,
):
    solver = EXPERT.SafeIK()
    elbow = EXPERT.JOINT_NAMES.index("elbow")
    solver.fk.lower[elbow] = np.deg2rad(EXPERT.ELBOW_LOWER_DEG)
    start, target, pick_yaw, place_yaw, _phases, attempts = EXPERT.sample_valid_episode(
        rng,
        solver,
        tolerance_mm=tolerance_mm,
        path_steps=path_steps,
        max_attempts=max_attempts,
    )
    waypoints = EXPERT.make_waypoints(start, target, pick_yaw, place_yaw)
    cosine, sine = np.cos(place_yaw), np.sin(place_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    world_compensation = rotation @ PHYSICS_PLACE_POSITION_COMPENSATION_LOCAL_M
    for name in ("place_approach", "place", "retreat"):
        waypoints[name][:2] -= world_compensation
    phases = EXPERT.plan_episode(
        solver,
        waypoints,
        pick_yaw,
        place_yaw - PHYSICS_PLACE_YAW_COMPENSATION_RAD,
        tolerance_mm=tolerance_mm,
        path_steps=path_steps,
    )
    return start, target, pick_yaw, place_yaw, phases, attempts


def record_episode(
    output_path: Path,
    *,
    episode_index: int,
    start: np.ndarray,
    target: np.ndarray,
    pick_yaw: float,
    place_yaw: float,
    phases,
    fps: int,
    width: int,
    height: int,
    max_place_error_mm: float = 35.0,
) -> dict[str, Any]:
    """물리 contact로 expert를 실행하며 RGB/state/action을 같은 시점에 기록한다."""
    model = EXPERT.load_physics_model()
    model.actuator_ctrlrange[model.actuator("elbow").id, 0] = np.deg2rad(
        EXPERT.ELBOW_LOWER_DEG
    )
    gripper_actuator = model.actuator("gripper_motor").id
    model.actuator_gainprm[gripper_actuator, 0] = 30.0
    model.actuator_biasprm[gripper_actuator, 1] = -30.0
    model.actuator_forcerange[gripper_actuator] = [-1.5, 1.5]
    for geom_name in (
        "gripper_left_collision", "gripper_right_collision",
        "gripper_right_part_collision",
    ):
        geom_id = model.geom(geom_name).id
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    arm_addresses = np.asarray([
        model.jnt_qposadr[model.joint(name).id] for name in EXPERT.JOINT_NAMES
    ])
    left_address = int(model.jnt_qposadr[model.joint("gripper_left").id])
    right_address = int(model.jnt_qposadr[model.joint("gripper_right").id])
    block_joint = model.joint("block_free").id
    block_address = int(model.jnt_qposadr[block_joint])
    tcp_site = model.site("graspframe").id
    target_body = model.body("place_target").id
    arm_actuators = np.asarray([model.actuator(name).id for name in EXPERT.JOINT_NAMES])
    block_geom = model.geom("block_geom").id
    left_pad_geom = model.geom("left_finger_pad_geom").id
    right_pad_geom = model.geom("right_finger_pad_geom").id
    model.body_pos[target_body] = [target[0], target[1], 0.001]
    model.body_quat[target_body] = [
        np.cos(place_yaw / 2), 0, 0, np.sin(place_yaw / 2)
    ]

    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    timestamps: list[float] = []
    phase_ids: list[int] = []
    tcp_positions: list[np.ndarray] = []
    block_poses: list[np.ndarray] = []
    def set_block(position: np.ndarray, yaw: float) -> None:
        data.qpos[block_address:block_address + 3] = position
        data.qpos[block_address + 3:block_address + 7] = [
            np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
        ]
    def finger_contacts() -> tuple[bool, bool]:
        pairs = {frozenset((int(contact.geom1), int(contact.geom2)))
                 for contact in data.contact}
        return (
            frozenset((block_geom, left_pad_geom)) in pairs,
            frozenset((block_geom, right_pad_geom)) in pairs,
        )

    def capture(phase_id: int, command: np.ndarray) -> None:
        renderer.update_scene(data, camera="dataset_camera")
        images.append(renderer.render().copy())
        states.append(np.r_[data.qpos[arm_addresses], data.qpos[left_address]])
        actions.append(command.copy())
        timestamps.append(len(timestamps) / fps)
        phase_ids.append(phase_id)
        tcp_positions.append(data.site_xpos[tcp_site].copy())
        quaternion = data.qpos[block_address + 3:block_address + 7]
        block_poses.append(np.r_[data.qpos[block_address:block_address + 3],
                                 yaw_from_quaternion(quaternion)])

    data.qpos[arm_addresses] = EXPERT.READY_Q
    data.qpos[left_address] = EXPERT.GRIPPER_OPEN
    data.qpos[right_address] = -EXPERT.GRIPPER_OPEN
    set_block(start, pick_yaw)
    data.qvel[:] = 0.0
    data.ctrl[arm_actuators] = EXPERT.READY_Q
    data.ctrl[gripper_actuator] = EXPERT.GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    phase_names = ["ready", "pick_approach", "pick", "grasp", "lift",
                   "place_approach", "place", "release", "retreat"]
    simulation_step = 0
    capture_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    max_block_height = float(data.qpos[block_address + 2])
    bilateral_grasp = False

    def command() -> np.ndarray:
        return np.r_[data.ctrl[arm_actuators], data.ctrl[gripper_actuator]]

    def run_segment(duration: float, phase_id: int, updater) -> None:
        nonlocal simulation_step, max_block_height
        steps = max(1, int(round(duration / model.opt.timestep)))
        for step in range(steps):
            updater(EXPERT.smoothstep((step + 1) / steps))
            mujoco.mj_step(model, data)
            simulation_step += 1
            max_block_height = max(max_block_height, float(data.qpos[block_address + 2]))
            if simulation_step % capture_every == 0:
                capture(phase_id, command())

    capture(phase_names.index("ready"), command())
    try:
        for phase in phases:
            phase_id = phase_names.index(phase.name)
            run_segment(
                phase.duration,
                phase_id,
                lambda alpha, item=phase: data.ctrl.__setitem__(
                    arm_actuators,
                    (1.0 - alpha) * item.start_q + alpha * item.goal_q,
                ),
            )
            if phase.name == "pick":
                initial = float(data.ctrl[gripper_actuator])
                run_segment(
                    2.0,
                    phase_names.index("grasp"),
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * initial + alpha * EXPERT.GRIPPER_CLOSED,
                    ),
                )
                run_segment(
                    EXPERT.GRASP_SETTLE_S,
                    phase_names.index("grasp"),
                    lambda _alpha: None,
                )
                bilateral_grasp = all(finger_contacts())
                if not bilateral_grasp:
                    raise RuntimeError("물리 파지 실패: 양쪽 finger pad가 블록에 접촉하지 않았습니다.")
            elif phase.name == "lift" and data.qpos[block_address + 2] < 0.08:
                raise RuntimeError("물리 파지 실패: 블록이 80mm 이상 들어 올려지지 않았습니다.")
            elif phase.name == "place":
                initial = float(data.ctrl[gripper_actuator])
                run_segment(
                    0.8,
                    phase_names.index("release"),
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * initial + alpha * EXPERT.GRIPPER_OPEN,
                    ),
                )
                run_segment(
                    EXPERT.RELEASE_SETTLE_S,
                    phase_names.index("release"),
                    lambda _alpha: None,
                )
    finally:
        renderer.close()

    final_position = data.qpos[block_address:block_address + 3].copy()
    final_yaw = yaw_from_quaternion(data.qpos[block_address + 3:block_address + 7])
    position_error_mm = float(np.linalg.norm(final_position - target) * 1000)
    yaw_error_deg = float(np.rad2deg(EXPERT.axis_error(final_yaw, place_yaw)))
    success = (
        bilateral_grasp
        and max_block_height >= 0.08
        and position_error_mm <= max_place_error_mm
    )
    if not success:
        raise RuntimeError(
            f"물리 episode 실패: 위치={position_error_mm:.3f}mm, "
            f"yaw={yaw_error_deg:.2f}°, 최고높이={max_block_height * 1000:.1f}mm"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=np.asarray(images, dtype=np.uint8),
        observation_state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        timestamp=np.asarray(timestamps, dtype=np.float64),
        phase=np.asarray(phase_ids, dtype=np.uint8),
        tcp_position=np.asarray(tcp_positions, dtype=np.float32),
        block_pose=np.asarray(block_poses, dtype=np.float32),
    )
    return {
        "episode_index": episode_index,
        "file": str(output_path.relative_to(output_path.parents[1])),
        "frames": len(images),
        "duration_s": timestamps[-1] if timestamps else 0.0,
        "pick_xyz_m": start.tolist(),
        "place_xyz_m": target.tolist(),
        "pick_yaw_rad": float(pick_yaw),
        "place_yaw_rad": float(place_yaw),
        "position_error_mm": position_error_mm,
        "yaw_error_deg": yaw_error_deg,
        "max_block_height_m": max_block_height,
        "bilateral_grasp_contact": bilateral_grasp,
        "physics_contact": True,
        "place_yaw_compensation_deg": float(
            np.rad2deg(PHYSICS_PLACE_YAW_COMPENSATION_RAD)
        ),
        "success": success,
    }


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "fps": None,
            "image_size": None,
            "joint_names": [*EXPERT.JOINT_NAMES, "gripper"],
            "features": FEATURES,
            "generation_mode": "mujoco_contact_physics",
            "episodes": [],
        }
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary = dataset_dir / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(dataset_dir / "manifest.json")
