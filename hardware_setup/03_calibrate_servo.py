#!/usr/bin/env python3
"""선택한 서보의 원점을 측정하고 robot.yaml에 저장합니다."""

from pathlib import Path
import shutil
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jdcobot200.hardware import MiniFeetechDriver  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "jdcobot200" / "robot.yaml"
PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000
LOGICAL_CENTER = 2048
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

print("=== 3. 서보 캘리브레이션 ===")
motor_id = int(input("캘리브레이션할 서보 ID (1~6): "))
if motor_id not in MOTOR_IDS:
    raise SystemExit("서보 ID는 1~6 중 하나여야 합니다.")

driver = MiniFeetechDriver(PORT, BAUDRATE)

try:
    driver.set_torque(motor_id, False)
    time.sleep(0.05)
    print(f"ID {motor_id} torque를 해제했습니다.")
    print("관절을 손으로 로봇의 원점 자세에 맞추세요.")
    input("정렬을 마쳤으면 Enter: ")

    raw = driver.get_position_filtered(motor_id, samples=7)
    if raw is None:
        raise SystemExit("엔코더 값을 읽지 못했습니다.")

    # robot.yaml 규약: raw home tick + home_offset_tick = 2048
    offset = LOGICAL_CENTER - raw
    print(f"raw home tick : {raw}")
    print(f"center tick   : {LOGICAL_CENTER}")
    print(f"새 offset     : {offset}")

    if input("robot.yaml에 저장하려면 SAVE 입력: ").strip() == "SAVE":
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        for joint in data["joints"]:
            if joint["motor_id"] == motor_id:
                joint["home_offset_tick"] = offset
                break

        backup = CONFIG_PATH.with_suffix(".yaml.bak")
        shutil.copy2(CONFIG_PATH, backup)
        CONFIG_PATH.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"저장 완료: {CONFIG_PATH}")
        print(f"이전 설정 백업: {backup}")
    else:
        print("측정값을 저장하지 않았습니다.")

    # 현재 위치를 목표로 기록한 후 다시 고정합니다.
    driver.set_position(motor_id, raw)
    driver.set_torque(motor_id, True)
    print(f"ID {motor_id}를 현재 위치에서 다시 고정했습니다.")
finally:
    driver.close()
