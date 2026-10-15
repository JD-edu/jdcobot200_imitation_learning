#!/usr/bin/env python3
"""블록 시작/목표 위치를 무작위화하여 여러 합성 PnP episode를 검증합니다."""

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
READY_Q = np.deg2rad([0.0, 20.0, -40.0, 20.0, 0.0])
ELBOW_LOWER_DEG = -110.0
BLOCK_Z = 0.025
TCP_GRASP_Z = 0.060
APPROACH_Z = 0.120
TRANSPORT_Z = 0.180
GRIPPER_OPEN = 0.60
GRIPPER_CLOSED = -0.06
GRASP_CENTER_OFFSET_LOCAL_M = np.array([0.0, -0.0045])
GRASP_SETTLE_S = 0.35
RELEASE_SETTLE_S = 0.50

# 로봇 정면의 pick 영역과 오른쪽의 place 영역을 분리하여 경로가 명확하게 보이게 한다.
PICK_X_RANGE = (-0.055, 0.055)
PICK_Y_RANGE = (0.155, 0.205)
PLACE_X_RANGE = (0.105, 0.175)
PLACE_Y_RANGE = (0.065, 0.135)


@dataclass(frozen=True)
class Phase:
    name: str
    start_q: np.ndarray
    goal_q: np.ndarray
    duration: float


class Viewer:
    def __init__(self, model: mujoco.MjModel) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW 초기화 실패: DISPLAY 설정을 확인하세요.")
        self.window = glfw.create_window(960, 720, "Randomized Synthetic PnP", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 창 생성 실패")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.08, 0.11, 0.15]
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


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def axis_error(first: float, second: float) -> float:
    return float(abs((first - second + np.pi / 2.0) % np.pi - np.pi / 2.0))


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def jaw_axis_yaw(solver: SafeIK, q_rad: np.ndarray) -> float:
    solver.fk.evaluate(q_rad)
    left = solver.model.body("gripper_left_assembly").id
    right = solver.model.body("gripper_right_assembly").id
    vector = solver.fk.data.xpos[right] - solver.fk.data.xpos[left]
    return float(np.arctan2(vector[1], vector[0]))


def align_wrist_roll(
    solver: SafeIK, q_rad: np.ndarray, block_yaw: float
) -> tuple[np.ndarray, float]:
    index = JOINT_NAMES.index("wrist_roll")
    desired_jaw_yaw = block_yaw + np.pi / 2.0
    best = None
    for wrist in np.linspace(solver.fk.lower[index], solver.fk.upper[index], 361):
        candidate = q_rad.copy()
        candidate[index] = wrist
        if not solver.fk.evaluate(candidate).safe:
            continue
        error = float(abs(
            (jaw_axis_yaw(solver, candidate) - desired_jaw_yaw + np.pi)
            % (2.0 * np.pi) - np.pi
        ))
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


def make_waypoints(
    start: np.ndarray, target: np.ndarray, pick_yaw: float, place_yaw: float
) -> dict[str, np.ndarray]:
    def corrected_xy(position: np.ndarray, yaw: float) -> np.ndarray:
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return position[:2] + rotation @ GRASP_CENTER_OFFSET_LOCAL_M

    pick_xy = corrected_xy(start, pick_yaw)
    place_xy = corrected_xy(target, place_yaw)
    return {
        "pick_approach": np.r_[pick_xy, APPROACH_Z],
        "pick": np.r_[pick_xy, TCP_GRASP_Z],
        "lift": np.r_[pick_xy, TRANSPORT_Z],
        "place_approach": np.r_[place_xy, TRANSPORT_Z],
        "place": np.r_[place_xy, TCP_GRASP_Z],
        "retreat": np.r_[place_xy, TRANSPORT_Z],
    }


def plan_episode(
    solver: SafeIK,
    waypoints: dict[str, np.ndarray],
    pick_yaw: float,
    place_yaw: float,
    *,
    tolerance_mm: float,
    path_steps: int,
) -> list[Phase]:
    durations = {
        "pick_approach": 2.5,
        "pick": 1.5,
        "lift": 2.0,
        "place_approach": 2.5,
        "place": 1.5,
        "retreat": 2.0,
    }
    phases = []
    current_q = READY_Q.copy()
    for name, target_xyz in waypoints.items():
        result = solver.solve(
            current_q, target_xyz, tolerance_m=tolerance_mm / 1000.0
        )
        if not result.converged:
            raise RuntimeError(
                f"{name} IK 미수렴(error={result.error_m * 1000:.2f}mm)"
            )
        desired_yaw = pick_yaw if name in {
            "pick_approach", "pick", "lift"
        } else place_yaw
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
                f"{name} 경로 unsafe(step={path.unsafe_step}, "
                f"reason={path.unsafe_result.reasons})"
            )
        phases.append(Phase(name, current_q.copy(), aligned_q.copy(), durations[name]))
        current_q = aligned_q
    return phases


def sample_valid_episode(
    rng: np.random.Generator,
    solver: SafeIK,
    *,
    tolerance_mm: float,
    path_steps: int,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, float, float, list[Phase], int]:
    """IK/FK 검사를 통과할 때까지 좌표를 다시 뽑는다."""
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        start = np.array([
            rng.uniform(*PICK_X_RANGE), rng.uniform(*PICK_Y_RANGE), BLOCK_Z
        ])
        target = np.array([
            rng.uniform(*PLACE_X_RANGE), rng.uniform(*PLACE_Y_RANGE), BLOCK_Z
        ])
        pick_yaw = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        place_yaw = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        try:
            phases = plan_episode(
                solver,
                make_waypoints(start, target, pick_yaw, place_yaw),
                pick_yaw,
                place_yaw,
                tolerance_mm=tolerance_mm,
                path_steps=path_steps,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
            print(f"샘플 거부 {attempt}: {last_error}")
            continue
        return start, target, pick_yaw, place_yaw, phases, attempt
    raise RuntimeError(
        f"{max_attempts}회 안에 안전한 episode를 찾지 못했습니다. "
        f"마지막 오류: {last_error}"
    )


def load_physics_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    pad_specs = (
        ("gripper_left_assembly", "left_finger_pad_geom",
         [-0.02108064, -0.05447028, 0.00712145],
         [0.35019978, 0.92846746, -0.12372670]),
        ("gripper_right_assembly", "right_finger_pad_geom",
         [-0.02589600, 0.05632086, 0.04163503],
         [0.62343708, -0.77202232, -0.12372447]),
    )
    for body_name, geom_name, position, z_axis in pad_specs:
        spec.body(body_name).add_geom(
            name=geom_name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=position, zaxis=z_axis, size=[0.008, 0.018, 0.0],
            mass=0.002, contype=1, conaffinity=1, condim=6,
            friction=[4.0, 0.30, 0.10], solref=[0.004, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0], margin=0.0005,
            rgba=[0.15, 0.15, 0.15, 1.0],
        )
    return spec.compile()


def simulate_episode(
    model: mujoco.MjModel,
    phases: list[Phase],
    block_start: np.ndarray,
    block_target: np.ndarray,
    pick_yaw: float,
    place_yaw: float,
    *,
    episode_index: int,
    fps: int,
    speed: float,
    use_viewer: bool,
) -> tuple[float, float, bool]:
    data = mujoco.MjData(model)
    arm_addresses = np.asarray([
        model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES
    ])
    left_address = int(model.jnt_qposadr[model.joint("gripper_left").id])
    right_address = int(model.jnt_qposadr[model.joint("gripper_right").id])
    block_joint = model.joint("block_free").id
    block_address = int(model.jnt_qposadr[block_joint])
    target_body = model.body("place_target").id
    model.body_pos[target_body] = [block_target[0], block_target[1], 0.001]
    model.body_quat[target_body] = [
        np.cos(place_yaw / 2.0), 0.0, 0.0, np.sin(place_yaw / 2.0)
    ]
    arm_actuators = np.asarray([model.actuator(name).id for name in JOINT_NAMES])
    gripper_actuator = model.actuator("gripper_motor").id

    def set_block(position: np.ndarray, yaw: float) -> None:
        data.qpos[block_address:block_address + 3] = position
        data.qpos[block_address + 3:block_address + 7] = [
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ]

    data.qpos[arm_addresses] = READY_Q
    data.qpos[left_address] = GRIPPER_OPEN
    data.qpos[right_address] = -GRIPPER_OPEN
    set_block(block_start, pick_yaw)
    data.qvel[:] = 0.0
    data.ctrl[arm_actuators] = READY_Q
    data.ctrl[gripper_actuator] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    viewer = Viewer(model) if use_viewer else None
    aborted = False

    def finger_contacts() -> tuple[bool, bool]:
        block_geom = model.geom("block_geom").id
        left_geom = model.geom("left_finger_pad_geom").id
        right_geom = model.geom("right_finger_pad_geom").id
        pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in data.contact}
        return (
            frozenset((block_geom, left_geom)) in pairs,
            frozenset((block_geom, right_geom)) in pairs,
        )

    def run_frames(label: str, duration: float, updater) -> None:
        nonlocal aborted
        count = max(1, int(round(duration / model.opt.timestep)))
        render_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
        for frame in range(count + 1):
            if viewer is not None and viewer.should_close():
                aborted = True
                return
            progress = smoothstep(frame / count)
            updater(progress)
            mujoco.mj_step(model, data)
            if viewer is not None and (frame % render_every == 0 or frame == count):
                viewer.render(
                    model, data,
                    f"Random PnP #{episode_index} | {label} | "
                    f"{progress * 100:5.1f}% | ESC: stop",
                )
                time.sleep(1.0 / (fps * speed))

    try:
        for phase in phases:
            run_frames(
                phase.name, phase.duration,
                lambda alpha, p=phase: data.ctrl.__setitem__(
                    arm_actuators, (1.0 - alpha) * p.start_q + alpha * p.goal_q
                ),
            )
            if aborted:
                break
            if phase.name == "lift" and data.qpos[block_address + 2] < 0.08:
                raise RuntimeError(
                    f"episode {episode_index} 파지 실패: 블록이 들어 올려지지 않았습니다."
                )
            if phase.name == "pick":
                initial = float(data.ctrl[gripper_actuator])
                run_frames(
                    "grasp", 2.0,
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * initial + alpha * GRIPPER_CLOSED,
                    ),
                )
                run_frames("grasp settle", GRASP_SETTLE_S, lambda _alpha: None)
                left_contact, right_contact = finger_contacts()
                print(
                    f"episode {episode_index} grasp 검사: "
                    f"left-contact={left_contact}, right-contact={right_contact}"
                )
                if not (left_contact and right_contact):
                    raise RuntimeError(
                        f"episode {episode_index} 파지 실패: 양손 접촉이 없습니다."
                    )
            elif phase.name == "place":
                initial = float(data.ctrl[gripper_actuator])
                run_frames(
                    "release", 0.8,
                    lambda alpha: data.ctrl.__setitem__(
                        gripper_actuator,
                        (1.0 - alpha) * initial + alpha * GRIPPER_OPEN,
                    ),
                )
                run_frames("release settle", RELEASE_SETTLE_S, lambda _alpha: None)
    finally:
        if viewer is not None:
            viewer.close()

    final_position = data.qpos[block_address:block_address + 3].copy()
    error_mm = float(np.linalg.norm(final_position - block_target) * 1000.0)
    final_yaw = quaternion_yaw(data.qpos[block_address + 3:block_address + 7])
    yaw_error_deg = float(np.rad2deg(axis_error(final_yaw, place_yaw)))
    return error_mm, yaw_error_deg, aborted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="무작위 블록/목표 위치에서 안전한 합성 PnP를 반복합니다."
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-sample-attempts", type=int, default=100)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--max-place-error-mm", type=float, default=35.0,
                        help="물리 파지 place 허용오차(기본 35 mm)")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1000:
        raise ValueError("episodes는 1~1000 범위여야 합니다.")
    if not 1 <= args.max_sample_attempts <= 10000:
        raise ValueError("max-sample-attempts는 1~10000 범위여야 합니다.")
    if not 0.1 <= args.speed <= 10 or not 10 <= args.fps <= 240:
        raise ValueError("speed는 0.1~10, fps는 10~240 범위여야 합니다.")
    if args.tolerance_mm <= 0 or args.max_place_error_mm <= 0 or args.path_steps < 2:
        raise ValueError("tolerance-mm은 양수, path-steps는 2 이상이어야 합니다.")

    solver = SafeIK()
    solver.fk.lower[JOINT_NAMES.index("elbow")] = np.deg2rad(ELBOW_LOWER_DEG)
    if not solver.fk.evaluate(READY_Q).safe:
        raise RuntimeError("READY 자세가 안전하지 않습니다.")
    model = load_physics_model()
    model.actuator_ctrlrange[model.actuator("elbow").id, 0] = np.deg2rad(
        ELBOW_LOWER_DEG
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
    rng = np.random.default_rng(args.random_seed)
    successes = 0
    total_attempts = 0

    for episode in range(1, args.episodes + 1):
        start, target, pick_yaw, place_yaw, phases, attempts = sample_valid_episode(
            rng,
            solver,
            tolerance_mm=args.tolerance_mm,
            path_steps=args.path_steps,
            max_attempts=args.max_sample_attempts,
        )
        total_attempts += attempts
        print(
            f"episode {episode}: pick(mm)={np.round(start * 1000, 2)}, "
            f"yaw={np.rad2deg(pick_yaw):.1f}°, "
            f"place(mm)={np.round(target * 1000, 2)}, "
            f"yaw={np.rad2deg(place_yaw):.1f}°, sampling={attempts}회"
        )
        error_mm, yaw_error_deg, aborted = simulate_episode(
            model,
            phases,
            start,
            target,
            pick_yaw,
            place_yaw,
            episode_index=episode,
            fps=args.fps,
            speed=args.speed,
            use_viewer=not args.no_viewer,
        )
        if aborted:
            print("사용자가 시뮬레이션을 중단했습니다.")
            return 130
        passed = error_mm <= args.max_place_error_mm
        successes += int(passed)
        print(
            f"episode {episode} {'성공' if passed else '실패'}: "
            f"place 오차={error_mm:.3f} mm, yaw 오차={yaw_error_deg:.2f}°"
        )

    print(
        f"완료: success={successes}/{args.episodes}, "
        f"안전 좌표 탐색 총 {total_attempts}회, seed={args.random_seed}"
    )
    if successes != args.episodes:
        raise RuntimeError("일부 randomized episode가 실패했습니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
