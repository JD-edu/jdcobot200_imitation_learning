#!/usr/bin/env python3
"""Print TCP XYZ while the torque-free JDCobot200 is moved by hand.

This is forward kinematics (encoder joint angles -> TCP XYZ), although it is
useful for collecting and checking targets later consumed by IK.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAFE_ROOT = REPO_ROOT / "7_FK_safe_zone"
HARDWARE_ROOT = REPO_ROOT / "9_robot_teaching_data_collection"
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
sys.path[:0] = [str(ROOT), str(SAFE_ROOT), str(HARDWARE_ROOT)]

from jdcobot200_safe_mink_ik import (  # noqa: E402
    CENTER_TICK,
    DEFAULT_JOINT_SIGNS,
    MOTOR_IDS,
    load_int_map,
    parse_signs,
    ticks_to_radians,
)
from motor_drive import MiniFeetechDriver  # noqa: E402
from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="손으로 로봇을 티칭하는 동안 엔코더 FK의 TCP XYZ를 출력합니다."
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--rate", type=float, default=5.0,
        help="터미널 갱신 주파수(Hz, 기본 5)",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="모터별 중앙값 필터 샘플 수(기본 1, 권장 최대 3)",
    )
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    parser.add_argument(
        "--one-line", action="store_true",
        help="같은 줄을 갱신하지 않고 매 측정값을 새 줄에 출력",
    )
    return parser.parse_args()


def read_ticks_once(
    driver: MiniFeetechDriver, samples: int
) -> tuple[np.ndarray | None, list[int]]:
    values = []
    failed = []
    for motor_id in MOTOR_IDS:
        if samples == 1:
            value = driver.get_position(int(motor_id))
        else:
            value = driver.get_position_filtered(int(motor_id), samples=samples)
        if value is None:
            failed.append(int(motor_id))
            values.append(0)
        else:
            values.append(int(value))
    if failed:
        return None, failed
    return np.asarray(values, dtype=int), failed


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("rate는 양수여야 합니다.")
    if not 1 <= args.samples <= 3:
        raise ValueError("samples는 통신 부하를 위해 1~3 범위로 지정하세요.")

    offsets = load_int_map(OFFSET_PATH)
    centers = np.array([
        CENTER_TICK + int(offsets[int(motor_id)]) for motor_id in MOTOR_IDS
    ])
    signs = parse_signs(args.joint_signs)
    kinematics = JDCobotSafetyModel()
    full_q = kinematics.configuration.q.copy()
    driver = MiniFeetechDriver(args.port, args.baudrate)

    try:
        print("모터 1~5의 토크를 해제합니다. 로봇 팔을 손으로 지지하십시오.")
        for motor_id in MOTOR_IDS:
            if not driver.set_torque_verified(int(motor_id), False):
                raise RuntimeError(f"모터 ID {motor_id} Torque OFF 확인 실패")
        print("Torque OFF 확인 완료")
        print("관절 순서:", ", ".join(ARM_JOINT_NAMES))
        print("단위: XYZ=mm, q=deg | 종료: Ctrl+C\n")

        period = 1.0 / args.rate
        next_sample = time.perf_counter()
        sequence = 0
        while True:
            ticks, failed = read_ticks_once(driver, args.samples)
            if ticks is None:
                text = f"COMM ERROR: encoder response missing from IDs {failed}"
            else:
                q = ticks_to_radians(ticks, centers, signs)
                # Do not use SafetyModel.evaluate here: it intentionally clips
                # to actuator limits. Teaching must report the actual encoder
                # posture even when it is outside the automatic-motion range.
                full_q[kinematics.qpos_addresses] = q
                kinematics.configuration.update(full_q)
                tcp_mm = (
                    kinematics.data.site_xpos[kinematics.site_id].copy() * 1000.0
                )
                sequence += 1
                text = (
                    f"#{sequence:06d}  "
                    f"XYZ(mm)=[{tcp_mm[0]:8.2f}, {tcp_mm[1]:8.2f}, {tcp_mm[2]:8.2f}]  "
                    f"q(deg)={np.array2string(np.rad2deg(q), precision=2, floatmode='fixed')}  "
                    f"ticks={ticks}"
                )
            if args.one_line:
                print(text, flush=True)
            else:
                print(f"\r{text:<180}", end="", flush=True)
            next_sample += period
            delay = next_sample - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_sample = time.perf_counter()
    except KeyboardInterrupt:
        if not args.one_line:
            print()
        print("티칭 XYZ 출력을 종료합니다. 모터 토크는 OFF 상태입니다.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
