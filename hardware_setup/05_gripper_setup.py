#!/usr/bin/env python3
"""변경된 그리퍼의 HOME, OPEN, CLOSE 위치를 다시 측정합니다."""

from pathlib import Path
import shutil
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jdcobot200.hardware import MiniFeetechDriver  # noqa: E402


PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000
GRIPPER_ID = 6
LOGICAL_CENTER = 2048
SAFE_MARGIN = 30
CONFIG_PATH = PROJECT_ROOT / "config" / "jdcobot200" / "robot.yaml"


def read_position(driver: MiniFeetechDriver, label: str) -> int:
    """사용자가 위치를 맞춘 뒤 안정된 encoder 값을 읽습니다."""
    input(f"그리퍼를 {label} 위치에 놓고 Enter: ")
    position = driver.get_position_filtered(GRIPPER_ID, samples=7)
    if position is None:
        raise RuntimeError(f"{label} 위치에서 encoder를 읽지 못했습니다.")
    print(f"{label}: {position} tick")
    return position


print("=== 5. Gripper Setup ===")
print("그리퍼 ID 6의 HOME, 기계적 OPEN, 기계적 CLOSE 위치를 측정합니다.")
print("손가락이나 기어에 힘을 주지 말고 끝 위치까지만 천천히 움직이세요.")
print(f"기계 끝단에서 {SAFE_MARGIN} tick 안쪽을 실제 명령 범위로 사용합니다.")

driver = MiniFeetechDriver(PORT, BAUDRATE)

try:
    driver.set_torque(GRIPPER_ID, False)
    time.sleep(0.05)
    print("ID 6 torque OFF: 이제 손으로 그리퍼를 움직일 수 있습니다.\n")

    home_raw = read_position(driver, "HOME")
    mechanical_open = read_position(driver, "완전 OPEN")
    mechanical_closed = read_position(driver, "완전 CLOSE")

    travel = abs(mechanical_open - mechanical_closed)
    if travel <= SAFE_MARGIN * 2:
        raise RuntimeError(
            f"측정 범위 {travel} tick이 너무 작습니다. OPEN/CLOSE를 다시 측정하세요."
        )

    # 각 기계 끝단에서 반대쪽 방향으로 SAFE_MARGIN만큼 이동합니다.
    toward_closed = 1 if mechanical_closed > mechanical_open else -1
    safe_open = mechanical_open + toward_closed * SAFE_MARGIN
    safe_closed = mechanical_closed - toward_closed * SAFE_MARGIN
    safe_min = min(safe_open, safe_closed)
    safe_max = max(safe_open, safe_closed)
    home_offset = LOGICAL_CENTER - home_raw

    if not safe_min <= home_raw <= safe_max:
        raise RuntimeError(
            f"HOME {home_raw}이 안전 범위 {safe_min}~{safe_max} 밖입니다."
        )

    print("\n=== 측정 결과 ===")
    print(f"HOME             : {home_raw}")
    print(f"기계 OPEN/CLOSE  : {mechanical_open} / {mechanical_closed}")
    print(f"안전 OPEN/CLOSE  : {safe_open} / {safe_closed}")
    print(f"안전 MIN/MAX     : {safe_min} / {safe_max}")
    print(f"HOME offset      : {home_offset}")

    if input("robot.yaml에 저장하려면 SAVE 입력: ").strip() != "SAVE":
        raise SystemExit("측정값을 저장하지 않았습니다.")

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gripper = next(
        joint for joint in data["joints"] if joint["motor_id"] == GRIPPER_ID
    )
    gripper["home_offset_tick"] = home_offset
    gripper["min_tick"] = safe_min
    gripper["max_tick"] = safe_max
    gripper["open_tick"] = safe_open
    gripper["closed_tick"] = safe_closed

    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, backup)
    CONFIG_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"저장 완료: {CONFIG_PATH}")
    print(f"이전 설정 백업: {backup}")

    # 설정 후 그리퍼가 떨어지거나 움직이지 않도록 측정한 HOME에서 고정합니다.
    driver.set_position(GRIPPER_ID, home_raw)
    driver.set_torque(GRIPPER_ID, True)
    print(f"ID 6을 HOME {home_raw}에서 고정했습니다.")
finally:
    driver.close()
