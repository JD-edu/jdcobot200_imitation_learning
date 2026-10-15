#!/usr/bin/env python3
"""ID 1~6 서보를 robot.yaml의 원점으로 천천히 이동합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/01_servo_homing_and_gripper/01_all_servo_homing.py

from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jdcobot200 import load_robot_config  # noqa: E402
from jdcobot200.hardware import MiniFeetechDriver  # noqa: E402


MOVE_TIME = 3.0
CONTROL_PERIOD = 0.02
POSITION_TOLERANCE = 15


def smoothstep(progress: float) -> float:
    """시작과 끝에서 속도가 부드럽게 0이 되는 S-curve입니다."""
    return progress * progress * (3.0 - 2.0 * progress)


config = load_robot_config()
driver = MiniFeetechDriver(config.port, config.baudrate)

try:
    start_positions = {}
    home_positions = {}

    print("=== 전체 서보 Homing: ID 1~6 ===")
    for joint in config.joints:
        current = driver.get_position_filtered(joint.motor_id, samples=5)
        if current is None:
            raise SystemExit(f"ID {joint.motor_id}({joint.name}) 위치 읽기 실패")

        home = config.logical_center_tick - joint.home_offset_tick
        if not joint.min_tick <= home <= joint.max_tick:
            raise SystemExit(
                f"ID {joint.motor_id} HOME {home}이 "
                f"안전 범위 {joint.min_tick}~{joint.max_tick} 밖입니다."
            )

        start_positions[joint.motor_id] = current
        home_positions[joint.motor_id] = home
        print(
            f"ID {joint.motor_id} {joint.name:12s}: "
            f"현재 {current:4d} -> HOME {home:4d}"
        )

    print("\n로봇팔을 지지하고 주변에 장애물이 없는지 확인하세요.")
    if input("전체 homing을 시작하려면 HOME 입력: ").strip() != "HOME":
        raise SystemExit("취소했습니다.")

    # Torque를 켰을 때 이전 목표로 튀지 않도록 현재 위치를 먼저 목표로 씁니다.
    for motor_id, current in start_positions.items():
        driver.set_position(motor_id, current)
    for motor_id in start_positions:
        driver.set_torque(motor_id, True)
        time.sleep(0.05)

    total_steps = int(MOVE_TIME / CONTROL_PERIOD)
    for step in range(total_steps + 1):
        alpha = smoothstep(step / total_steps)
        for motor_id in start_positions:
            start = start_positions[motor_id]
            home = home_positions[motor_id]
            command = round(start + (home - start) * alpha)
            driver.set_position(motor_id, command)
            time.sleep(0.002)
        time.sleep(CONTROL_PERIOD)

    print("\n=== Homing 결과 ===")
    all_reached = True
    for joint in config.joints:
        measured = driver.get_position_filtered(joint.motor_id, samples=3)
        target = home_positions[joint.motor_id]
        error = None if measured is None else abs(measured - target)
        reached = error is not None and error <= POSITION_TOLERANCE
        all_reached = all_reached and reached
        print(
            f"ID {joint.motor_id}: 목표={target}, 측정={measured}, "
            f"오차={error}, {'OK' if reached else 'FAIL'}"
        )

    if not all_reached:
        raise SystemExit("일부 서보가 HOME에 도달하지 못했습니다.")

    print("전체 서보가 HOME에 도달했습니다.")
    print("자세 유지를 위해 torque는 ON 상태로 남깁니다.")
finally:
    driver.close()
