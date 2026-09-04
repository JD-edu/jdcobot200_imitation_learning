#!/usr/bin/env python3
"""Move the real JDCobot200 only after a MINK-FK safe-zone path check."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
HARDWARE_ROOT = REPO_ROOT / "1_jdcobot200_hardware_control"
SAFE_ZONE_ROOT = ROOT
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = HARDWARE_ROOT / "joint_limits.txt"

sys.path.insert(0, str(HARDWARE_ROOT))
sys.path.insert(0, str(SAFE_ZONE_ROOT))

from motor_control import MiniFeetechDriver  # noqa: E402
from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel  # noqa: E402


MOTOR_IDS = np.arange(1, 6, dtype=int)
CENTER_TICK = 2048
TICKS_PER_REVOLUTION = 4096.0
# Motor 4 rotates opposite to the MuJoCo wrist_pitch positive direction.
DEFAULT_JOINT_SIGNS = "1,1,1,-1,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MINK FK safe-zone 검사 후 실제 JDCobot200을 이동합니다."
    )
    parser.add_argument(
        "--goal-deg", nargs=5, type=float, required=True,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="MuJoCo 좌표계 기준 목표 관절각 5개(도)",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--check-steps", type=int, default=201)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    parser.add_argument(
        "--release-after-move", action="store_true",
        help="이동 완료 후 모터 토크 해제(팔이 처질 수 있으므로 주의)",
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
    values = []
    for motor_id in MOTOR_IDS:
        value = driver.get_position(int(motor_id))
        if value is None:
            raise RuntimeError(f"모터 ID {motor_id} 위치 읽기 실패")
        values.append(value)
    return np.asarray(values, dtype=int)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> int:
    args = parse_args()
    if args.move_time <= 0 or args.check_steps < 2:
        raise ValueError("--move-time은 양수, --check-steps는 2 이상이어야 합니다.")
    if args.collision_margin_mm < 0 or args.floor_margin_mm < 0:
        raise ValueError("안전 margin은 0 이상이어야 합니다.")

    signs = parse_signs(args.joint_signs)
    offsets = load_int_map(OFFSET_PATH)
    limits = load_int_map(LIMIT_PATH, pair=True)
    centers = np.array([CENTER_TICK + int(offsets[int(i)]) for i in MOTOR_IDS])
    lower_ticks = np.array([
        limits[int(i)][0] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)
    upper_ticks = np.array([
        limits[int(i)][1] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)
    goal_q = np.deg2rad(np.asarray(args.goal_deg, dtype=float))
    safety = JDCobotSafetyModel()

    driver = MiniFeetechDriver(args.port, args.baudrate)
    torque_enabled = False
    try:
        start_ticks = read_positions(driver)
        if np.any(start_ticks < lower_ticks) or np.any(start_ticks > upper_ticks):
            raise RuntimeError(
                f"현재 엔코더가 하드웨어 안전범위 밖입니다: {start_ticks}"
            )
        start_q = ticks_to_radians(start_ticks, centers, signs)
        goal_ticks = radians_to_ticks(goal_q, centers, signs)
        if np.any(goal_ticks < lower_ticks) or np.any(goal_ticks > upper_ticks):
            raise RuntimeError(
                "목표가 하드웨어 joint_limits.txt 범위 밖입니다.\n"
                f"목표 tick={goal_ticks}, 최소={lower_ticks}, 최대={upper_ticks}"
            )

        trajectory = np.linspace(start_q, goal_q, args.check_steps)
        collision_margin = args.collision_margin_mm / 1000.0
        floor_margin = args.floor_margin_mm / 1000.0
        worst_self = (float("inf"), 0, ("", ""))
        worst_floor = (float("inf"), 0, "")
        first_unsafe = None
        start_result = None
        goal_result = None
        for index, q in enumerate(trajectory):
            result = safety.evaluate(q, collision_margin, floor_margin)
            if index == 0:
                start_result = result
            if index == len(trajectory) - 1:
                goal_result = result
            if result.minimum_self_clearance < worst_self[0]:
                worst_self = (
                    result.minimum_self_clearance, index, result.closest_self_pair
                )
            if result.minimum_floor_clearance < worst_floor[0]:
                worst_floor = (
                    result.minimum_floor_clearance, index, result.closest_floor_geom
                )
            if not result.safe and first_unsafe is None:
                first_unsafe = (index, q.copy(), result.reasons)

        assert start_result is not None and goal_result is not None
        print("관절 순서:", ", ".join(ARM_JOINT_NAMES))
        print("현재 q(deg):", np.round(np.rad2deg(start_q), 3))
        print("목표 q(deg):", np.round(np.rad2deg(goal_q), 3))
        print("현재 TCP(m):", np.round(start_result.tcp_position, 5))
        print("목표 TCP(m):", np.round(goal_result.tcp_position, 5))
        print(
            f"경로 최소 self clearance: {worst_self[0] * 1000:.2f} mm "
            f"(step {worst_self[1]}, {worst_self[2][0]} / {worst_self[2][1]})"
        )
        print(
            f"경로 최소 floor clearance: {worst_floor[0] * 1000:.2f} mm "
            f"(step {worst_floor[1]}, {worst_floor[2]})"
        )
        if first_unsafe is not None:
            index, q, reasons = first_unsafe
            print(
                "UNSAFE: 이동을 차단했습니다. "
                f"step={index}, q(deg)={np.round(np.rad2deg(q), 3)}, "
                f"reason={','.join(reasons)}"
            )
            return 2

        print(f"SAFE: 전체 {args.check_steps}개 경로 자세가 margin을 만족합니다.")
        print("목표 tick:", goal_ticks)
        if input("실제 로봇을 움직이려면 MOVE를 입력하세요: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        for motor_id in MOTOR_IDS:
            driver.set_torque(int(motor_id), True)
            time.sleep(0.03)
        torque_enabled = True

        started = time.perf_counter()
        while True:
            progress = (time.perf_counter() - started) / args.move_time
            alpha = smoothstep(progress)
            command_q = (1.0 - alpha) * start_q + alpha * goal_q
            command_ticks = radians_to_ticks(command_q, centers, signs)
            for motor_id, tick in zip(MOTOR_IDS, command_ticks):
                driver.set_position(int(motor_id), int(tick))
            if progress >= 1.0:
                break
            time.sleep(0.05)

        measured_ticks = read_positions(driver)
        measured_q = ticks_to_radians(measured_ticks, centers, signs)
        final_result = safety.evaluate(measured_q, collision_margin, floor_margin)
        print("이동 완료 q(deg):", np.round(np.rad2deg(measured_q), 3))
        print("이동 완료 TCP(m):", np.round(final_result.tcp_position, 5))
        print("목표 오차(deg):", np.round(np.rad2deg(measured_q - goal_q), 3))
        if not args.release_after_move:
            print("목표 자세 유지를 위해 모터 토크를 켠 상태로 종료합니다.")
        return 0
    finally:
        if torque_enabled and args.release_after_move:
            for motor_id in MOTOR_IDS:
                try:
                    driver.set_torque(int(motor_id), False)
                except Exception:
                    pass
        driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
