#!/usr/bin/env python3
"""선택한 서보 또는 전체 서보의 torque를 해제합니다."""

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

print("=== 2. 서보 릴리스 ===")
print("경고: torque를 해제하면 로봇팔이 아래로 떨어질 수 있습니다.")
print("로봇팔을 손으로 지지한 후 진행하세요.")
selected = int(input("해제할 서보 ID (1~6, 전체는 0): "))

if selected == 0:
    targets = MOTOR_IDS
elif selected in MOTOR_IDS:
    targets = [selected]
else:
    raise SystemExit("서보 ID는 0~6 중 하나여야 합니다.")

if input("torque를 해제하려면 RELEASE 입력: ").strip() != "RELEASE":
    raise SystemExit("취소했습니다.")

driver = MiniFeetechDriver(PORT, BAUDRATE)

try:
    for motor_id in targets:
        driver.set_torque(motor_id, False)
        time.sleep(0.02)
        print(f"ID {motor_id}: torque OFF")
finally:
    driver.close()
