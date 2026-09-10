#!/usr/bin/env python3
"""MINK position IK with simulation, safe-zone, and real-robot validation.

Pipeline:
  encoder -> calibrated MuJoCo q -> MINK IK -> limits -> path safe-zone
  -> FK error -> MOVE confirmation -> minimum-jerk motion -> encoder FK check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mink
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAFE_ROOT = REPO_ROOT / "7_FK_safe_zone"
HARDWARE_ROOT = REPO_ROOT / "9_robot_teaching_data_collection"
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = REPO_ROOT / "1_jdcobot200_hardware_control" / "joint_limits.txt"

sys.path.insert(0, str(SAFE_ROOT))
sys.path.insert(0, str(HARDWARE_ROOT))
from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel  # noqa: E402
from motor_drive import MiniFeetechDriver  # noqa: E402


MOTOR_IDS = np.arange(1, 6, dtype=int)
CENTER_TICK = 2048
TICKS_PER_REVOLUTION = 4096.0
DEFAULT_JOINT_SIGNS = "1,1,1,-1,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MINK IK 결과를 검증한 뒤 실제 JDCobot200을 이동합니다."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="로봇 기준 절대 TCP 위치(m)",
    )
    target.add_argument(
        "--delta-xyz-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
        help="현재 TCP 기준 상대 이동(mm); 최초 시험에 권장",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--simulation-only", action="store_true",
        help="시리얼 연결 없이 MuJoCo에서만 검사하고 경로 재생",
    )
    parser.add_argument(
        "--start-q-deg", nargs=5, type=float, default=[0, 0, 0, 0, 0],
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="simulation-only 시작 관절각(도, 기본값: 모두 0)",
    )
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    parser.add_argument("--ik-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--ik-iterations", type=int, default=500)
    parser.add_argument(
        "--max-joint-change-deg", type=float,
        help="1회 최대 관절 변화량(도); 기본값: 실제 로봇 10, 시뮬레이션 30",
    )
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    parser.add_argument(
        "--release-after-move", action="store_true",
        help="완료 후 토크 해제(팔이 처질 수 있으므로 주의)",
    )
    return parser.parse_args()


def load_int_map(path: Path, pair: bool = False) -> dict[int, int | tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"필수 설정 파일이 없습니다: {path}")
    values: dict[int, int | tuple[int, int]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            key_text, value_text = line.split("=", 1)
            if pair:
                low_text, high_text = value_text.split(",", 1)
                values[int(key_text)] = (int(low_text), int(high_text))
            else:
                values[int(key_text)] = int(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} 형식 오류: {line}") from exc
    missing = [int(i) for i in MOTOR_IDS if int(i) not in values]
    if missing:
        raise ValueError(f"{path}에 모터 ID가 없습니다: {missing}")
    return values


def parse_signs(text: str) -> np.ndarray:
    try:
        signs = np.asarray([int(value) for value in text.split(",")], dtype=int)
    except ValueError as exc:
        raise ValueError("--joint-signs는 1 또는 -1 다섯 개여야 합니다.") from exc
    if signs.shape != (5,) or not np.all(np.isin(signs, (-1, 1))):
        raise ValueError("--joint-signs는 1 또는 -1 다섯 개여야 합니다.")
    return signs


def ticks_to_radians(ticks: np.ndarray, centers: np.ndarray,
                     signs: np.ndarray) -> np.ndarray:
    return signs * (ticks - centers) * (2.0 * np.pi / TICKS_PER_REVOLUTION)


def radians_to_ticks(q: np.ndarray, centers: np.ndarray,
                     signs: np.ndarray) -> np.ndarray:
    return np.rint(
        centers + signs * q * (TICKS_PER_REVOLUTION / (2.0 * np.pi))
    ).astype(int)


def read_positions(driver: MiniFeetechDriver) -> np.ndarray:
    positions = []
    for motor_id in MOTOR_IDS:
        value = driver.get_position_filtered(int(motor_id), samples=3)
        if value is None:
            raise RuntimeError(f"모터 ID {motor_id} 위치 읽기 실패")
        positions.append(value)
    return np.asarray(positions, dtype=int)


def solve_position_ik(safety: JDCobotSafetyModel, start_q: np.ndarray,
                      target_xyz: np.ndarray, iterations: int,
                      tolerance_m: float) -> tuple[np.ndarray, float, int]:
    """Solve position-only IK while keeping all non-arm joints fixed."""
    model = safety.model
    configuration = mink.Configuration(model)
    full_q = np.zeros(model.nq, dtype=float)
    full_q[safety.qpos_addresses] = start_q
    configuration.update(full_q)

    frame_task = mink.FrameTask(
        frame_name="graspframe", frame_type="site",
        position_cost=1.0, orientation_cost=0.0, lm_damping=1e-3,
    )
    frame_task.set_target(mink.SE3.from_translation(target_xyz))
    posture_task = mink.PostureTask(model=model, cost=1e-3)
    posture_task.set_target_from_configuration(configuration)
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(
            model=model,
            velocities={name: 1.0 for name in ARM_JOINT_NAMES},
        ),
    ]
    dof_addresses = model.jnt_dofadr[safety.joint_ids].astype(int)
    site_id = model.site("graspframe").id
    previous_error = float("inf")
    stagnant_steps = 0

    for iteration in range(1, iterations + 1):
        error = float(np.linalg.norm(target_xyz - configuration.data.site_xpos[site_id]))
        if error <= tolerance_m:
            break
        try:
            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=[frame_task, posture_task], dt=0.02,
                solver="daqp", limits=limits, damping=1e-6,
            )
        except mink.NoSolutionFound as exc:
            raise RuntimeError("제한 조건을 만족하는 IK 해를 찾지 못했습니다.") from exc

        arm_velocity = np.zeros(model.nv, dtype=float)
        arm_velocity[dof_addresses] = velocity[dof_addresses]
        configuration.integrate_inplace(arm_velocity, 0.02)
        clipped = np.clip(
            configuration.q[safety.qpos_addresses], safety.lower, safety.upper
        )
        configuration.q[safety.qpos_addresses] = clipped
        configuration.update(configuration.q)

        stagnant_steps = stagnant_steps + 1 if abs(previous_error - error) < 1e-8 else 0
        previous_error = error
        if stagnant_steps >= 30:
            break

    solution_q = configuration.q[safety.qpos_addresses].copy()
    final_error = float(np.linalg.norm(target_xyz - configuration.data.site_xpos[site_id]))
    return solution_q, final_error, iteration


def check_safe_path(safety: JDCobotSafetyModel, start_q: np.ndarray,
                    goal_q: np.ndarray, steps: int, collision_margin: float,
                    floor_margin: float):
    worst_self = (float("inf"), 0, ("", ""))
    worst_floor = (float("inf"), 0, "")
    first_unsafe = None
    for index, q in enumerate(np.linspace(start_q, goal_q, steps)):
        result = safety.evaluate(q, collision_margin, floor_margin)
        if result.minimum_self_clearance < worst_self[0]:
            worst_self = (result.minimum_self_clearance, index, result.closest_self_pair)
        if result.minimum_floor_clearance < worst_floor[0]:
            worst_floor = (result.minimum_floor_clearance, index, result.closest_floor_geom)
        if not result.safe and first_unsafe is None:
            first_unsafe = (index, q.copy(), result.reasons)
    return first_unsafe, worst_self, worst_floor


def minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value ** 3 * (10.0 - 15.0 * value + 6.0 * value ** 2)


def play_mujoco_trajectory(safety: JDCobotSafetyModel, start_q: np.ndarray,
                           goal_q: np.ndarray, duration: float) -> None:
    """Kinematically replay the verified path in a GLFW window."""
    from mujoco.glfw import glfw

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패: DISPLAY와 그래픽 드라이버를 확인하세요.")

    window = glfw.create_window(1200, 900, "JDCobot200 MINK IK simulation", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW 창 생성 실패")

    try:
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        scene = mujoco.MjvScene(safety.model, maxgeom=10_000)
        context = mujoco.MjrContext(
            safety.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.22]
        camera.distance = 0.8
        camera.azimuth = 145.0
        camera.elevation = -25.0
        option = mujoco.MjvOption()

        def key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(window, key_callback)
        full_q = safety.configuration.q.copy()
        started = time.perf_counter()
        completed_at = None

        while not glfw.window_should_close(window):
            now = time.perf_counter()
            progress = min(1.0, (now - started) / duration)
            alpha = minimum_jerk(progress)
            full_q[safety.qpos_addresses] = (
                (1.0 - alpha) * start_q + alpha * goal_q
            )
            # FK-only playback: collision proxies are queried but their contact
            # forces are not integrated into the displayed joint motion.
            safety.configuration.update(full_q)

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                safety.model, safety.data, option, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()

            glfw.set_window_title(
                window,
                f"JDCobot200 MINK IK | progress={progress * 100:5.1f}% | ESC: close",
            )
            if progress >= 1.0:
                if completed_at is None:
                    completed_at = now
                elif now - completed_at >= 1.0:
                    break
            time.sleep(1.0 / 120.0)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> int:
    args = parse_args()
    if args.move_time <= 0 or args.path_steps < 2 or args.ik_iterations < 1:
        raise ValueError("move-time은 양수, path-steps는 2 이상이어야 합니다.")
    if min(args.collision_margin_mm, args.floor_margin_mm,
           args.ik_tolerance_mm) < 0:
        raise ValueError("margin, tolerance, 최대 관절 변화량은 0 이상이어야 합니다.")
    if args.max_joint_change_deg is not None and args.max_joint_change_deg < 0:
        raise ValueError("최대 관절 변화량은 0 이상이어야 합니다.")
    max_joint_change_deg = (
        args.max_joint_change_deg
        if args.max_joint_change_deg is not None
        else (30.0 if args.simulation_only else 10.0)
    )

    safety = JDCobotSafetyModel()
    signs = parse_signs(args.joint_signs)
    driver = None
    torque_enabled = False
    centers = lower_ticks = upper_ticks = start_ticks = None

    try:
        # 1. Select the simulation start pose or read the real encoders.
        if args.simulation_only:
            start_q = np.deg2rad(np.asarray(args.start_q_deg, dtype=float))
            if np.any(start_q < safety.lower) or np.any(start_q > safety.upper):
                raise RuntimeError("simulation 시작 자세가 MuJoCo 관절 제한 밖입니다.")
        else:
            offsets = load_int_map(OFFSET_PATH)
            limits = load_int_map(LIMIT_PATH, pair=True)
            centers = np.array([
                CENTER_TICK + int(offsets[int(i)]) for i in MOTOR_IDS
            ])
            lower_ticks = np.array([
                limits[int(i)][0] + int(offsets[int(i)])  # type: ignore[index]
                for i in MOTOR_IDS
            ], dtype=int)
            upper_ticks = np.array([
                limits[int(i)][1] + int(offsets[int(i)])  # type: ignore[index]
                for i in MOTOR_IDS
            ], dtype=int)
            driver = MiniFeetechDriver(args.port, args.baudrate)
            start_ticks = read_positions(driver)
            if np.any(start_ticks < lower_ticks) or np.any(start_ticks > upper_ticks):
                raise RuntimeError(f"현재 엔코더가 하드웨어 안전범위 밖입니다: {start_ticks}")
            start_q = ticks_to_radians(start_ticks, centers, signs)
        start_result = safety.evaluate(start_q)
        if not start_result.safe:
            raise RuntimeError(f"현재 자세가 safe-zone 밖입니다: {start_result.reasons}")

        if args.target_xyz is not None:
            target_xyz = np.asarray(args.target_xyz, dtype=float)
        else:
            target_xyz = start_result.tcp_position + (
                np.asarray(args.delta_xyz_mm, dtype=float) / 1000.0
            )

        # 2. Position-only MINK IK.
        goal_q, ik_error, used_iterations = solve_position_ik(
            safety, start_q, target_xyz, args.ik_iterations,
            args.ik_tolerance_mm / 1000.0,
        )
        if ik_error > args.ik_tolerance_mm / 1000.0:
            raise RuntimeError(
                f"IK가 허용 오차에 수렴하지 않았습니다: {ik_error * 1000:.2f} mm"
            )

        # 3. Simulation and hardware joint limits.
        joint_change = np.rad2deg(goal_q - start_q)
        if np.any(np.abs(joint_change) > max_joint_change_deg):
            raise RuntimeError(
                "IK 관절 변화량이 1회 이동 제한을 초과했습니다.\n"
                f"제한={max_joint_change_deg:g} deg, "
                f"변화량(deg)={np.round(joint_change, 3)}"
            )
        goal_ticks = None
        if not args.simulation_only:
            goal_ticks = radians_to_ticks(goal_q, centers, signs)
            if np.any(goal_ticks < lower_ticks) or np.any(goal_ticks > upper_ticks):
                raise RuntimeError(f"IK 목표 tick이 하드웨어 제한 밖입니다: {goal_ticks}")

        # 4. Check every interpolated configuration against the safe-zone.
        collision_margin = args.collision_margin_mm / 1000.0
        floor_margin = args.floor_margin_mm / 1000.0
        unsafe, worst_self, worst_floor = check_safe_path(
            safety, start_q, goal_q, args.path_steps,
            collision_margin, floor_margin,
        )
        if unsafe is not None:
            index, q, reasons = unsafe
            raise RuntimeError(
                f"경로가 safe-zone을 벗어납니다: step={index}, "
                f"q(deg)={np.round(np.rad2deg(q), 3)}, reason={reasons}"
            )

        # 5. Independent FK check of the final simulated configuration.
        goal_result = safety.evaluate(goal_q, collision_margin, floor_margin)
        fk_error = float(np.linalg.norm(goal_result.tcp_position - target_xyz))
        if fk_error > args.ik_tolerance_mm / 1000.0:
            raise RuntimeError(f"시뮬레이션 FK 검증 실패: {fk_error * 1000:.2f} mm")

        print("\nMINK IK + SAFE-ZONE 사전 검사 통과")
        print("관절 순서:", ", ".join(ARM_JOINT_NAMES))
        print("현재 TCP(m):", np.round(start_result.tcp_position, 5))
        print("목표 TCP(m):", np.round(target_xyz, 5))
        print("IK TCP(m):  ", np.round(goal_result.tcp_position, 5))
        print(f"IK/FK 오차: {fk_error * 1000:.3f} mm ({used_iterations} iterations)")
        print("현재 q(deg):", np.round(np.rad2deg(start_q), 3))
        print("목표 q(deg):", np.round(np.rad2deg(goal_q), 3))
        print("변화량(deg):", np.round(joint_change, 3))
        print(f"1회 관절 변화 제한: {max_joint_change_deg:g} deg")
        print(f"경로 최소 self clearance: {worst_self[0] * 1000:.2f} mm")
        print(f"경로 최소 floor clearance: {worst_floor[0] * 1000:.2f} mm")
        if goal_ticks is not None:
            print("목표 tick:", goal_ticks)
        if args.simulation_only:
            print("SIMULATION ONLY: 실제 로봇에는 연결하거나 명령하지 않습니다.")
            play_mujoco_trajectory(safety, start_q, goal_q, args.move_time)
            print("MuJoCo 경로 재생을 완료했습니다.")
            return 0
        if input("실제 로봇을 움직이려면 MOVE를 입력하세요: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        # 6. Minimum-jerk real-robot motion.
        for motor_id in MOTOR_IDS:
            driver.set_position(int(motor_id), int(start_ticks[motor_id - 1]))
            if not driver.set_torque_verified(int(motor_id), True):
                raise RuntimeError(f"모터 ID {motor_id} Torque ON 확인 실패")
            time.sleep(0.05)
        torque_enabled = True
        started = time.perf_counter()
        next_update = started
        last_sent = np.full(5, -1, dtype=int)
        while True:
            progress = min(1.0, (time.perf_counter() - started) / args.move_time)
            alpha = minimum_jerk(progress)
            command_q = (1.0 - alpha) * start_q + alpha * goal_q
            command_ticks = radians_to_ticks(command_q, centers, signs)
            for index, (motor_id, tick) in enumerate(zip(MOTOR_IDS, command_ticks)):
                if tick != last_sent[index]:
                    driver.set_position(int(motor_id), int(tick))
                    last_sent[index] = tick
            if progress >= 1.0:
                break
            next_update += 0.025
            time.sleep(max(0.0, next_update - time.perf_counter()))

        # 7. Encoder -> MINK FK post-check.
        time.sleep(0.3)
        measured_ticks = read_positions(driver)
        measured_q = ticks_to_radians(measured_ticks, centers, signs)
        measured_result = safety.evaluate(measured_q, collision_margin, floor_margin)
        real_fk_error = float(np.linalg.norm(measured_result.tcp_position - target_xyz))
        print("\n이동 완료")
        print("실측 q(deg):", np.round(np.rad2deg(measured_q), 3))
        print("실측 TCP(m):", np.round(measured_result.tcp_position, 5))
        print(f"실제 엔코더 FK 오차: {real_fk_error * 1000:.2f} mm")
        print("도착 safe-zone:", "SAFE" if measured_result.safe else measured_result.reasons)
        if not args.release_after_move:
            print("목표 자세 유지를 위해 토크를 켠 상태로 종료합니다.")
        return 0
    finally:
        if torque_enabled and args.release_after_move:
            for motor_id in MOTOR_IDS:
                try:
                    driver.set_torque_verified(int(motor_id), False)
                except Exception:
                    pass
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
