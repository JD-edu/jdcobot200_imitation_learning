#!/usr/bin/env python3
"""선택한 서보를 10 tick만 움직였다가 원위치로 복귀합니다."""

import time

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jdcobot200.hardware import MiniFeetechDriver  # noqa: E402


PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "jdcobot200" / "robot.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

print("=== 4. 서보 미소 이동 ===")
motor_id = int(input("이동할 서보 ID (1~6): "))
joint = next((item for item in config["joints"] if item["motor_id"] == motor_id), None)
if joint is None:
    raise SystemExit("서보 ID는 1~6 중 하나여야 합니다.")

direction = int(input("이동 방향 (+1 또는 -1): "))
if direction not in (-1, 1):
    raise SystemExit("방향은 +1 또는 -1이어야 합니다.")

driver = MiniFeetechDriver(PORT, BAUDRATE)

try:
    start = driver.get_position_filtered(motor_id, samples=5)
    if start is None:
        raise SystemExit("엔코더 값을 읽지 못했습니다.")

    target = start + direction * 10
    if not joint["min_tick"] <= target <= joint["max_tick"]:
        raise SystemExit(
            f"목표 {target}이 안전 범위 "
            f"{joint['min_tick']}~{joint['max_tick']} 밖입니다."
        )

    print(f"ID {motor_id}: {start} -> {target} -> {start}")
    print("10 tick은 약 0.88도입니다.")
    if input("이동하려면 MOVE 입력: ").strip() != "MOVE":
        raise SystemExit("취소했습니다.")

    driver.set_torque(motor_id, True)
    time.sleep(0.05)

    driver.set_position(motor_id, target)
    time.sleep(1.0)
    reached = driver.get_position_filtered(motor_id, samples=3)
    print(f"이동 후 위치: {reached}")

    driver.set_position(motor_id, start)
    time.sleep(1.0)
    returned = driver.get_position_filtered(motor_id, samples=3)
    print(f"복귀 후 위치: {returned}")
finally:
    driver.set_torque(motor_id, False)
    print(f"ID {motor_id}: torque OFF")
    driver.close()
