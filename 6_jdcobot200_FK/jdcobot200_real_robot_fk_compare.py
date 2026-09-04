#!/usr/bin/env python3
"""Compare relative real-robot FK with a synchronized MuJoCo simulation.

The test moves one arm joint at a time by a small angle and returns it to the
startup pose. The startup encoders and MuJoCo q=0 are treated as the same
relative origin. Hardware motion is impossible until the operator types MOVE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
from mujoco.glfw import glfw
import numpy as np

from jdcobot200_FK_mink_demo import CameraController, require_id, smoothstep
from jdcobot200_mink_fk_solver import SO101MinkFKSolver


ROOT = Path(__file__).resolve().parent
HARDWARE_ROOT = ROOT.parent / "1_jdcobot200_hardware_control"
sys.path.insert(0, str(HARDWARE_ROOT))
from motor_control import MiniFeetechDriver  # noqa: E402


XML_PATH = ROOT / "scene.xml"
OFFSET_PATH = ROOT.parent / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = HARDWARE_ROOT / "joint_limits.txt"
MOTOR_IDS = np.arange(1, 6, dtype=int)
CENTER_TICK = 2048
TICKS_PER_REVOLUTION = 4096.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="실제 JDCobot200 엔코더 FK와 MuJoCo FK를 작은 각도로 비교합니다."
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--angle-deg", type=float, default=2.0,
                        help="관절별 상대 시험각(기본값: 2도, 최대: 5도)")
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--status-period", type=float, default=0.25,
                        help="터미널 상태 출력 주기(초)")
    parser.add_argument("--joint", type=int, choices=range(1, 6),
                        help="지정한 한 관절만 시험(기본값: 1~5 순서대로)")
    parser.add_argument(
        "--joint-signs", default="1,1,1,1,1",
        help="모터 양의 방향과 MJCF 축 방향 매핑(쉼표로 구분한 5개 ±1)",
    )
    return parser.parse_args()


def load_int_map(path: Path, pair: bool = False) -> dict[int, int | tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"필수 설정 파일이 없습니다: {path}")
    values: dict[int, int | tuple[int, int]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
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
    missing = [int(motor_id) for motor_id in MOTOR_IDS if int(motor_id) not in values]
    if missing:
        raise ValueError(f"{path}에 모터 ID가 없습니다: {missing}")
    return values


def parse_signs(text: str) -> np.ndarray:
    try:
        signs = np.array([int(value) for value in text.split(",")], dtype=int)
    except ValueError as exc:
        raise ValueError("--joint-signs는 1 또는 -1 다섯 개여야 합니다.") from exc
    if signs.shape != (5,) or not np.all(np.isin(signs, (-1, 1))):
        raise ValueError("--joint-signs는 1 또는 -1 다섯 개여야 합니다.")
    return signs


def ticks_to_radians(raw_ticks: np.ndarray, centers: np.ndarray,
                     signs: np.ndarray) -> np.ndarray:
    return signs * (raw_ticks - centers) * (2.0 * np.pi / TICKS_PER_REVOLUTION)


def radians_to_ticks(q: np.ndarray, centers: np.ndarray,
                     signs: np.ndarray) -> np.ndarray:
    return np.rint(centers + signs * q * (TICKS_PER_REVOLUTION / (2.0 * np.pi))).astype(int)


def read_positions(driver: MiniFeetechDriver) -> np.ndarray:
    positions = []
    for motor_id in MOTOR_IDS:
        value = driver.get_position(int(motor_id))
        if value is None:
            raise RuntimeError(f"모터 ID {motor_id} 위치 읽기 실패")
        positions.append(value)
    return np.asarray(positions, dtype=int)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.angle_deg <= 5.0:
        raise ValueError("--angle-deg는 0보다 크고 5도 이하여야 합니다.")
    if args.move_time <= 0.0 or args.hold_time < 0.0 or args.status_period <= 0.0:
        raise ValueError("이동 시간은 양수, 유지 시간은 0 이상이어야 합니다.")

    offsets = load_int_map(OFFSET_PATH)
    limits = load_int_map(LIMIT_PATH, pair=True)
    signs = parse_signs(args.joint_signs)
    centers = np.array([CENTER_TICK + int(offsets[int(i)]) for i in MOTOR_IDS])
    # joint_limits.txt was recorded as ``raw tick - calibration offset``.
    # Convert it back to raw servo ticks before checking reads and commands.
    lower_ticks = np.array([
        limits[int(i)][0] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)
    upper_ticks = np.array([
        limits[int(i)][1] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    fk = SO101MinkFKSolver(model, "graspframe")
    joint_ids = np.array([model.joint(name).id for name in fk.ARM_JOINT_NAMES])
    qpos_addresses = model.jnt_qposadr[joint_ids].astype(int)
    actuator_ids = np.array([model.actuator(name).id for name in fk.ARM_JOINT_NAMES])
    gripper_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    site_id = model.site("graspframe").id

    driver = MiniFeetechDriver(args.port, args.baudrate)
    try:
        start_ticks = read_positions(driver)
        if np.any(start_ticks < lower_ticks) or np.any(start_ticks > upper_ticks):
            raise RuntimeError(
                "현재 엔코더가 joint_limits.txt 안전범위 밖입니다.\n"
                f"현재={start_ticks}, 최소={lower_ticks}, 최대={upper_ticks}"
            )
        # Keep the calibrated absolute angle only as a diagnostic. For this
        # comparison, the encoder position observed at startup is relative
        # q=0, exactly matching MuJoCo's qpos=0 initial configuration.
        calibrated_start_q = ticks_to_radians(start_ticks, centers, signs)
        relative_zero_q = np.zeros(5, dtype=float)
        selected = [args.joint - 1] if args.joint else list(range(5))
        delta = np.deg2rad(args.angle_deg)
        targets = []
        labels = []
        for index in selected:
            moved = relative_zero_q.copy()
            moved[index] += delta
            targets.extend((moved, relative_zero_q.copy()))
            labels.extend((f"J{index + 1} +{args.angle_deg:g} deg", "START RETURN"))
        # The actual startup ticks, rather than 2048+offset, are the origin
        # used to generate every relative hardware command.
        target_ticks = [radians_to_ticks(q, start_ticks, signs) for q in targets]
        for label, ticks in zip(labels, target_ticks):
            if np.any(ticks < lower_ticks) or np.any(ticks > upper_ticks):
                raise RuntimeError(f"{label} 목표가 안전범위 밖입니다: {ticks}")

        print("\n실제 로봇 미세각 FK 비교 테스트")
        print(f"포트: {args.port}, 시험각: {args.angle_deg:g} deg")
        print(f"상대 원점 tick: {start_ticks}")
        print(f"calibration 기준 잔차 q(rad): {np.round(calibrated_start_q, 5)}")
        print("MuJoCo 시작 q(rad): [0. 0. 0. 0. 0.]")
        for label, ticks in zip(labels, target_ticks):
            print(f"  {label:16s}: target={ticks}, delta={ticks - start_ticks} tick")
        print("동작: " + " -> ".join(labels))
        print("주변에 사람/장애물이 없고 비상 정지가 가능한지 확인하세요.")
        if input("실제 모터를 움직이려면 MOVE를 입력하세요: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        for motor_id in MOTOR_IDS:
            driver.set_torque(int(motor_id), True)
            time.sleep(0.03)

        mujoco.mj_resetData(model, data)
        data.qpos[qpos_addresses] = relative_zero_q
        data.ctrl[actuator_ids] = relative_zero_q
        data.ctrl[gripper_id] = 0.0
        mujoco.mj_forward(model, data)

        if not glfw.init():
            raise RuntimeError("GLFW 초기화 실패: DISPLAY 설정을 확인하세요.")
        window = glfw.create_window(1200, 900, "Real robot / MuJoCo FK", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("GLFW 창 생성 실패")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        scene = mujoco.MjvScene(model, maxgeom=10_000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        camera = CameraController(model, scene)
        glfw.set_mouse_button_callback(window, camera.mouse_button)
        glfw.set_cursor_pos_callback(window, camera.cursor_position)
        glfw.set_scroll_callback(window, camera.scroll)
        glfw.set_key_callback(
            window,
            lambda window, key, scancode, action, mods:
            glfw.set_window_should_close(window, True)
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE else None,
        )

        target_index = 0
        segment_start = time.perf_counter()
        segment_start_q = relative_zero_q.copy()
        holding = False
        next_hardware_update = 0.0
        next_status_time = 0.0
        measured_ticks = start_ticks.copy()
        maximum_encoder_delta = np.zeros(5, dtype=int)
        try:
            while not glfw.window_should_close(window) and target_index < len(targets):
                now = time.perf_counter()
                if not holding:
                    progress = (now - segment_start) / args.move_time
                    alpha = smoothstep(progress)
                    command_q = (1.0 - alpha) * segment_start_q + alpha * targets[target_index]
                    if progress >= 1.0:
                        command_q = targets[target_index].copy()
                        holding, segment_start = True, now
                else:
                    command_q = targets[target_index]
                    if now - segment_start >= args.hold_time:
                        segment_start_q = command_q.copy()
                        target_index += 1
                        if target_index >= len(targets):
                            break
                        holding, segment_start = False, now

                command_ticks = radians_to_ticks(command_q, start_ticks, signs)
                if now >= next_hardware_update:
                    for motor_id, tick in zip(MOTOR_IDS, command_ticks):
                        driver.set_position(int(motor_id), int(tick))
                    measured_ticks = read_positions(driver)
                    maximum_encoder_delta = np.maximum(
                        maximum_encoder_delta, np.abs(measured_ticks - start_ticks)
                    )
                    next_hardware_update = now + 0.05

                data.ctrl[actuator_ids] = command_q
                sim_end = data.time + 1.0 / 60.0
                while data.time < sim_end:
                    mujoco.mj_step(model, data)

                measured_q = ticks_to_radians(measured_ticks, start_ticks, signs)
                real_qpos = data.qpos.copy()
                real_qpos[qpos_addresses] = measured_q
                real_fk = fk.solve_step(real_qpos)
                sim_xyz = data.site_xpos[site_id].copy()
                difference_mm = np.linalg.norm(real_fk.position - sim_xyz) * 1000.0
                label = labels[target_index]
                if now >= next_status_time:
                    print(
                        f"[{label:16s} {('HOLD' if holding else 'MOVE'):4s}] "
                        f"cmd={command_ticks} encoder={measured_ticks} "
                        f"delta={measured_ticks - start_ticks} "
                        f"FK_diff={difference_mm:.2f} mm"
                    )
                    next_status_time = now + args.status_period
                status = "HOLD" if holding else "MOVE"
                glfw.set_window_title(
                    window, f"{label} | {status} | FK difference {difference_mm:.2f} mm"
                )
                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    model, data, camera.option, None, camera.camera,
                    mujoco.mjtCatBit.mjCAT_ALL.value, scene
                )
                mujoco.mjr_render(viewport, scene, context)
                overlay = (
                    f"{label} [{status}]\n"
                    f"Command q: {np.array2string(command_q, precision=4)}\n"
                    f"Encoder q: {np.array2string(measured_q, precision=4)}\n"
                    f"Real encoder FK: {np.array2string(real_fk.position, precision=5)}\n"
                    f"MuJoCo FK:      {np.array2string(sim_xyz, precision=5)}\n"
                    f"Difference: {difference_mm:.2f} mm"
                )
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport, overlay, "Esc: stop", context
                )
                glfw.swap_buffers(window)
                glfw.poll_events()
        finally:
            # Stop means hold the measured pose; never unexpectedly release a loaded arm.
            try:
                final_ticks = read_positions(driver)
                for motor_id, tick in zip(MOTOR_IDS, final_ticks):
                    driver.set_position(int(motor_id), int(tick))
            finally:
                context.free()
                scene.free()
                glfw.destroy_window(window)
                glfw.terminate()

        requested_tick_delta = max(
            int(np.max(np.abs(ticks - start_ticks))) for ticks in target_ticks
        )
        print(f"최대 명령 변화: {requested_tick_delta} tick")
        print(f"최대 encoder 변화: {maximum_encoder_delta} tick")
        for index in selected:
            if maximum_encoder_delta[index] < max(1, requested_tick_delta // 2):
                print(
                    f"경고: J{index + 1} encoder가 명령을 충분히 추종하지 않았습니다. "
                    "서보 deadband/마찰, 토크 상태 및 배선을 확인하세요."
                )
        print("테스트 완료. 실제 로봇은 마지막 위치에서 토크를 유지합니다.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n사용자 중단: 현재 위치 유지 명령 후 종료합니다.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise
