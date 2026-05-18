import time
from motor_control import MiniFeetechDriver


if __name__ == "__main__":
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    DEG_TO_TICK = 4096.0 / 360.0
    # 좌우로 각각 10도씩 움직이도록 설정 (총 20도 가동 범위)
    SWING_DEG = 10 
    SWING_TICK = int(SWING_DEG * DEG_TO_TICK)

    STEP_TICK = 2
    STEP_DELAY = 0.03
    HOLD_TIME = 0.3

    try:
        print("[1] 현재 위치를 기준 위치로 읽습니다.")

        center_positions = {}

        for motor_id in MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=5)

            if pos is None:
                print(f"ID {motor_id}: 위치 읽기 실패")
                driver.close()
                exit()

            center_positions[motor_id] = pos
            driver.set_torque(motor_id, True)
            time.sleep(0.05)

        print("기준 위치:", center_positions)
        print("[2] 각 관절을 '차례대로' 현재 위치 기준 ±10도씩 움직입니다.")
        print("Ctrl+C로 종료")

      
        # 모든 모터를 순서대로 하나씩 제어합니다.
        for motor_id in MOTOR_IDS:
            print(f"-> 현재 구동 중인 모터 ID: {motor_id}")

            # 1. 중심 -> +10도 방향 이동
            for step in range(0, SWING_TICK + 1, STEP_TICK):
                target = center_positions[motor_id] + step
                target = max(0, min(4095, target))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # 2. +10도 -> -10도 방향 이동
            for step in range(SWING_TICK, -SWING_TICK - 1, -STEP_TICK):
                target = center_positions[motor_id] + step
                target = max(0, min(4095, target))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # 3. -10도 -> 다시 중심(0도)으로 복귀
            for step in range(-SWING_TICK, 1, STEP_TICK):
                target = center_positions[motor_id] + step
                target = max(0, min(4095, target))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME) # 한 모터의 작동이 끝나면 잠시 대기 후 다음 모터로 이동

    except KeyboardInterrupt:
        print("\n종료합니다.")

    finally:
        driver.close()