#!/usr/bin/env python3
"""선택한 서보를 현재 위치에서 고정합니다."""

import time
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jdcobot200.hardware import MiniFeetechDriver  # noqa: E402


PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

print("=== 1. 서보 프리즈 ===")
print("관절이 갑자기 움직이지 않도록 현재 위치를 먼저 읽습니다.")
motor_id = int(input("고정할 서보 ID (1~6): "))

if motor_id not in MOTOR_IDS:
    raise SystemExit("서보 ID는 1~6 중 하나여야 합니다.")

driver = MiniFeetechDriver(PORT, BAUDRATE)

try:
    current = driver.get_position_filtered(motor_id, samples=5)
    if current is None:
        raise SystemExit("엔코더 값을 읽지 못했습니다.")

    print(f"ID {motor_id} 현재 위치: {current}")
    driver.set_position(motor_id, current)
    time.sleep(0.05)
    driver.set_torque(motor_id, True)
    time.sleep(0.1)
    print(f"ID {motor_id}를 현재 위치 {current}에서 고정했습니다.")
finally:
    driver.close()
