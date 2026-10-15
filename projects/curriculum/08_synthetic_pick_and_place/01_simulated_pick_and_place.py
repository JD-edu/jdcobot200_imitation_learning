#!/usr/bin/env python3
"""고정된 블록을 IK expert로 옮기는 첫 MuJoCo 합성 PnP 예제."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import mujoco
from mujoco.glfw import glfw
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parent
COURSE03 = PROJECT_ROOT / "projects/curriculum/03_mujoco_fk_safety"
COURSE05 = PROJECT_ROOT / "projects/curriculum/05_mujoco_ik"
for path in (COURSE03, COURSE05):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fk_safety import JOINT_NAMES  # noqa: E402
from ik_solver import SafeIK  # noqa: E402


SCENE_PATH = LESSON_DIR / "scenes/fixed_pick_and_place.xml"
READY_Q_DEG = np.array([0.0, 20.0, -40.0, 20.0, 0.0])
ELBOW_LOWER_DEG = -110.0
BLOCK_START_M = np.array([0.0, 0.18, 0.025])
BLOCK_TARGET_M = np.array([0.14, 0.10, 0.025])
BLOCK_START_YAW = np.deg2rad(30.0)
BLOCK_TARGET_YAW = np.deg2rad(-25.0)
# CAD의 graspframe은 두 실제 fingertip 접촉면의 중심에서 약 4.5 mm 벗어나 있다.
# 블록 좌표계의 짧은 축 방향 보정값이며 yaw에 맞춰 회전해 사용한다.
GRASP_CENTER_OFFSET_LOCAL_M = np.array([0.0, -0.0045])
GRIPPER_OPEN = 0.60
GRIPPER_CLOSED = -0.06
GRASP_SETTLE_S = 0.35
RELEASE_SETTLE_S = 0.50

def grasp_center_offset(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]]) @ GRASP_CENTER_OFFSET_LOCAL_M


PICK_XY = BLOCK_START_M[:2] + grasp_center_offset(BLOCK_START_YAW)
PLACE_XY = BLOCK_TARGET_M[:2] + grasp_center_offset(BLOCK_TARGET_YAW)
TCP_WAYPOINTS_M = {
    "pick_approach": np.r_[PICK_XY, 0.12],
    "pick": np.r_[PICK_XY, 0.06],
    "lift": np.r_[PICK_XY, 0.18],
    "place_approach": np.r_[PLACE_XY, 0.18],
    "place": np.r_[PLACE_XY, 0.06],
    "retreat": np.r_[PLACE_XY, 0.18],
}


@dataclass(frozen=True)
class ArmPhase:
    name: str
    start_q: np.ndarray
    goal_q: np.ndarray
    duration: float


def axis_error(first: float, second: float) -> float:
    """방향성이 없는 직육면체 축의 최소 각도 오차(0~pi/2)."""
    return float(abs((first - second + np.pi / 2.0) % np.pi - np.pi / 2.0))


def jaw_axis_yaw(solver: SafeIK, q_rad: np.ndarray) -> float:
    solver.fk.evaluate(q_rad)
    left = solver.model.body("gripper_left_assembly").id
    right = solver.model.body("gripper_right_assembly").id
    vector = solver.fk.data.xpos[right] - solver.fk.data.xpos[left]
    return float(np.arctan2(vector[1], vector[0]))


def align_wrist_roll(
    solver: SafeIK, q_rad: np.ndarray, block_yaw: float
) -> tuple[np.ndarray, float]:
    """손가락 닫힘 축이 블록 긴 축에 수직이 되도록 wrist-roll을 탐색한다."""
    index = JOINT_NAMES.index("wrist_roll")
    # 실제 두 손가락 형상은 180도 회전 대칭이 아니므로 방향까지 고정한다.
    desired_jaw_yaw = block_yaw + np.pi / 2.0
    best = None
    for wrist in np.linspace(solver.fk.lower[index], solver.fk.upper[index], 361):
        candidate = q_rad.copy()
        candidate[index] = wrist
        result = solver.fk.evaluate(candidate)
        if not result.safe:
            continue
        error = float(abs(
            (jaw_axis_yaw(solver, candidate) - desired_jaw_yaw + np.pi)
            % (2.0 * np.pi) - np.pi
        ))
        # 직육면체 축은 180도 대칭이므로 0.5도 이내 후보 중 현재 wrist와 가장
        # 가까운 값을 골라 불필요한 180도 회전을 방지한다.
        score = (
            round(error / np.deg2rad(0.5)),
            abs(wrist - q_rad[index]),
            error,
        )
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        raise RuntimeError("안전한 wrist-roll 정렬 해가 없습니다.")
    return best[1], best[0][2]


class Viewer:
    def __init__(self, model: mujoco.MjModel) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW 초기화 실패: DISPLAY 설정을 확인하세요.")
        self.window = glfw.create_window(960, 720, "Synthetic Pick-and-Place", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 창 생성 실패")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.08, 0.10, 0.16]
        self.camera.distance = 0.72
        self.camera.azimuth = 145.0
        self.camera.elevation = -28.0
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10_000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        glfw.set_key_callback(
            self.window,
            lambda window, key, _scan, action, _mods:
            glfw.set_window_should_close(window, True)
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS else None,
        )

    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.window))

    def render(self, model: mujoco.MjModel, data: mujoco.MjData, title: str) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            model, data, self.option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.set_window_title(self.window, title)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        glfw.destroy_window(self.window)
        glfw.terminate()


def smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def plan_ik(solver: SafeIK, path_steps: int, tolerance_mm: float) -> list[ArmPhase]:
    """모든 waypoint의 IK와 전체 경로를 실행 전에 계산한다."""
    solver.fk.lower[JOINT_NAMES.index("elbow")] = np.deg2rad(ELBOW_LOWER_DEG)
    current_q = np.deg2rad(READY_Q_DEG)
    initial = solver.fk.evaluate(current_q)
    if not initial.safe:
        raise RuntimeError(f"READY 자세가 안전하지 않습니다: {initial.reasons}")

    durations = {
        "pick_approach": 2.5,
        "pick": 1.5,
        "lift": 2.0,
        "place_approach": 2.5,
        "place": 1.5,
        "retreat": 2.0,
    }
    phases = []
    for name, target_xyz in TCP_WAYPOINTS_M.items():
        result = solver.solve(
            current_q, target_xyz, tolerance_m=tolerance_mm / 1000.0
        )
        if not result.converged:
            raise RuntimeError(
                f"{name} IK 미수렴: FK 오차 {result.error_m * 1000:.3f} mm"
            )
        desired_yaw = BLOCK_START_YAW if name in {
            "pick_approach", "pick", "lift"
        } else BLOCK_TARGET_YAW
        aligned_q, yaw_error = align_wrist_roll(solver, result.q_rad, desired_yaw)
        goal = solver.fk.evaluate(aligned_q)
        if not goal.safe:
            raise RuntimeError(f"{name} 목표 FK unsafe: {goal.reasons}")
        position_error = float(np.linalg.norm(goal.tcp_position - target_xyz))
        if position_error > tolerance_mm / 1000.0:
            raise RuntimeError(f"{name} wrist 정렬 후 위치 오차 초과")
        path = solver.check_path(current_q, aligned_q, steps=path_steps)
        if not path.safe:
            raise RuntimeError(
                f"{name} 경로 unsafe: step={path.unsafe_step}/{path_steps - 1}, "
                f"reason={path.unsafe_result.reasons}"
            )
        phases.append(ArmPhase(name, current_q.copy(), aligned_q.copy(), durations[name]))
        print(
            f"계획 {name}: target(mm)={np.round(target_xyz * 1000, 1)}, "
            f"error={position_error * 1000:.3f}mm, "
            f"yaw-error={np.rad2deg(yaw_error):.2f}deg, "
            f"q(deg)={np.round(np.rad2deg(aligned_q), 2)}"
        )
        current_q = aligned_q
    return phases


def set_block(data: mujoco.MjData, block_address: int, position: np.ndarray,
              yaw: float) -> None:
    data.qpos[block_address:block_address + 3] = position
    data.qpos[block_address + 3:block_address + 7] = [
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    ]


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def finger_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[bool, bool]:
    """현재 contact 목록에서 블록-왼/오른 손가락 접촉을 찾는다."""
    block_geom = model.geom("block_geom").id
    left_geom = model.geom("left_finger_pad_geom").id
    right_geoms = {model.geom("right_finger_pad_geom").id}
    pairs = {
        frozenset((int(contact.geom1), int(contact.geom2)))
        for contact in data.contact
    }
    left = frozenset((block_geom, left_geom)) in pairs
    right = any(frozenset((block_geom, geom)) in pairs for geom in right_geoms)
    return left, right


def load_physics_model() -> mujoco.MjModel:
    """CAD 손가락 body에 단순하고 안정적인 고무 pad collision을 직접 추가한다."""
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    pad_specs = (
        (
            "gripper_left_assembly", "left_finger_pad_geom",
            [-0.02108064, -0.05447028, 0.00712145],
            [0.35019978, 0.92846746, -0.12372670],
        ),
        (
            "gripper_right_assembly", "right_finger_pad_geom",
            [-0.02589600, 0.05632086, 0.04163503],
            [0.62343708, -0.77202232, -0.12372447],
        ),
    )
    for body_name, geom_name, position, z_axis in pad_specs:
        spec.body(body_name).add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=position,
            zaxis=z_axis,
            size=[0.008, 0.018, 0.0],
            mass=0.002,
            contype=1,
            conaffinity=1,
            condim=6,
            friction=[4.0, 0.30, 0.10],
            solref=[0.004, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            margin=0.0005,
            rgba=[0.15, 0.15, 0.15, 1.0],
        )
    return spec.compile()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="고정된 블록을 IK expert로 옮기는 MuJoCo 합성 PnP 예제"
    )
    parser.add_argument("--speed", type=float, default=1.0,
                        help="재생 속도 배율(기본 1, 허용 0.1~5)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--no-viewer", action="store_true",
                        help="창과 실시간 대기 없이 자동 검증")
    args = parser.parse_args()
    if not 0.1 <= args.speed <= 5.0 or not 10 <= args.fps <= 240:
        raise ValueError("speed는 0.1~5, fps는 10~240 범위여야 합니다.")
    if args.tolerance_mm <= 0 or args.path_steps < 2:
        raise ValueError("tolerance-mm은 양수, path-steps는 2 이상이어야 합니다.")

    solver = SafeIK()
    phases = plan_ik(solver, args.path_steps, args.tolerance_mm)
    model = load_physics_model()
    data = mujoco.MjData(model)

    arm_addresses = np.asarray([
        model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES
    ])
    left_address = int(model.jnt_qposadr[model.joint("gripper_left").id])
    right_address = int(model.jnt_qposadr[model.joint("gripper_right").id])
    arm_actuators = np.asarray([model.actuator(name).id for name in JOINT_NAMES])
    gripper_actuator = model.actuator("gripper_motor").id
    # 이 과제의 IK 안전범위(-110 deg)와 actuator의 제어범위를 일치시킨다.
    model.actuator_ctrlrange[model.actuator("elbow").id, 0] = np.deg2rad(
        ELBOW_LOWER_DEG
    )
    # 작은 물체를 고출력 위치 서보로 튕겨내지 않도록 그리퍼만 compliance를 준다.
    model.actuator_gainprm[gripper_actuator, 0] = 30.0
    model.actuator_biasprm[gripper_actuator, 1] = -30.0
    model.actuator_forcerange[gripper_actuator] = [-1.5, 1.5]
    # 부정확한 STL convex hull 대신 scene의 측면 fingertip pad를 접촉에 쓴다.
    for geom_name in (
        "gripper_left_collision",
        "gripper_right_collision",
        "gripper_right_part_collision",
    ):
        geom_id = model.geom(geom_name).id
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    block_joint_id = model.joint("block_free").id
    block_address = int(model.jnt_qposadr[block_joint_id])
    tcp_site_id = model.site("graspframe").id
    target_body_id = model.body("place_target").id
    model.body_quat[target_body_id] = [
        np.cos(BLOCK_TARGET_YAW / 2.0), 0.0, 0.0,
        np.sin(BLOCK_TARGET_YAW / 2.0),
    ]

    ready_q = np.deg2rad(READY_Q_DEG)
    data.qpos[arm_addresses] = ready_q
    data.qpos[left_address] = GRIPPER_OPEN
    data.qpos[right_address] = -GRIPPER_OPEN
    set_block(data, block_address, BLOCK_START_M, BLOCK_START_YAW)
    data.qvel[:] = 0.0
    data.ctrl[arm_actuators] = ready_q
    data.ctrl[gripper_actuator] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    viewer = None if args.no_viewer else Viewer(model)
    aborted = False

    def run_frames(label: str, duration: float, update) -> None:
        nonlocal aborted
        steps = max(1, int(round(duration / model.opt.timestep)))
        render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
        for index in range(steps + 1):
            if viewer is not None and viewer.should_close():
                aborted = True
                return
            progress = index / steps
            update(smoothstep(progress))
            mujoco.mj_step(model, data)
            if viewer is not None and (index % render_every == 0 or index == steps):
                viewer.render(
                    model, data,
                    f"Synthetic PnP | {label} | {progress * 100:5.1f}% | ESC: stop",
                )
                time.sleep(1.0 / (args.fps * args.speed))

    try:
        for phase in phases:
            if aborted:
                break
            run_frames(
                phase.name,
                phase.duration,
                lambda alpha, p=phase: data.ctrl.__setitem__(
                    arm_actuators,
                    (1.0 - alpha) * p.start_q + alpha * p.goal_q,
                ),
            )
            if aborted:
                break
            if phase.name in {"lift", "place_approach"}:
                print(
                    f"{phase.name} 후 블록(mm): "
                    f"{np.round(data.qpos[block_address:block_address + 3] * 1000, 2)}"
                )
            if phase.name == "lift" and data.qpos[block_address + 2] < 0.08:
                raise RuntimeError("파지 실패: 블록이 마찰로 들어 올려지지 않았습니다.")
            if phase.name == "pick":
                start_value = float(data.ctrl[gripper_actuator])
                run_frames(
                    "grasp",
                    2.0,
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * start_value + alpha * GRIPPER_CLOSED,
                    ),
                )
                run_frames("grasp settle", GRASP_SETTLE_S, lambda _alpha: None)
                left_contact, right_contact = finger_contacts(model, data)
                contact_names = [
                    (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
                    for contact in data.contact
                ]
                print(
                    f"물리 접촉 검사: left={left_contact}, right={right_contact}, "
                    f"finger_q={data.qpos[left_address]:.3f}, "
                    f"tcp(mm)={np.round(data.site_xpos[tcp_site_id] * 1000, 2)}, "
                    f"block(mm)={np.round(data.qpos[block_address:block_address + 3] * 1000, 2)}, "
                    f"contacts={contact_names}"
                )
                if not (left_contact and right_contact):
                    raise RuntimeError("파지 실패: 블록이 양쪽 손가락과 접촉하지 않았습니다.")
            elif phase.name == "place":
                start_value = float(data.ctrl[gripper_actuator])
                run_frames(
                    "release",
                    0.8,
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * start_value + alpha * GRIPPER_OPEN,
                    ),
                )
                run_frames("release settle", RELEASE_SETTLE_S, lambda _alpha: None)
    finally:
        if viewer is not None:
            viewer.close()

    if aborted:
        print("사용자가 시뮬레이션을 중단했습니다.")
        return 130

    final_block = data.qpos[block_address:block_address + 3].copy()
    error_mm = float(np.linalg.norm(final_block - BLOCK_TARGET_M) * 1000.0)
    final_yaw = quaternion_yaw(data.qpos[block_address + 3:block_address + 7])
    yaw_error_deg = float(np.rad2deg(axis_error(final_yaw, BLOCK_TARGET_YAW)))
    print("블록 시작 위치(mm):", np.round(BLOCK_START_M * 1000.0, 3))
    print("블록 목표 위치(mm):", np.round(BLOCK_TARGET_M * 1000.0, 3))
    print("블록 최종 위치(mm):", np.round(final_block * 1000.0, 3))
    print(f"place 오차: {error_mm:.3f} mm")
    print(
        f"블록 최종/목표 yaw: {np.rad2deg(final_yaw):.2f}/"
        f"{np.rad2deg(BLOCK_TARGET_YAW):.2f}°, 오차={yaw_error_deg:.2f}°"
    )
    if error_mm > 5.0:
        raise RuntimeError(
            f"합성 PnP 실패: place 오차 {error_mm:.3f} mm"
        )
    print("합성 PnP 성공")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
