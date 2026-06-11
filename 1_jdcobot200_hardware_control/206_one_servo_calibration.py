import time
from motor_control import MiniFeetechDriver

def calibrate_single_servo():
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    LOGICAL_CENTER = 2048

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        print("=" * 70)
        print("STS3215 Single Homing Offset Calibration")
        print("=" * 70)

        # 사용자로부터 칼리브레이션할 서보 ID 입력 받기
        target_id = int(input("교정할 서보 모터 ID를 입력하세요 (예: 1~6): "))
       
        print(f"\n[1] ID {target_id}번 모터 토크 OFF")
        # 선택한 단일 모터만 토크를 해제합니다. (나머지 모터 상태 유지)
        driver.set_torque(target_id, False)
        time.sleep(0.03)

        print(f"\nID {target_id}번 관절을 손으로 원하는 중심 자세에 맞추세요.")
        input("자세를 맞춘 후 [Enter]를 누르세요... ")

        print("\n[2] 현재 엔코더값 기준으로 offset 계산")
        print("-" * 70)

        # 현재 필터링된 엔코더 위치 읽기
        raw_pos = driver.get_position_filtered(target_id, samples=7)

        if raw_pos is None:
            print(f"ID {target_id}: 엔코더 읽기 실패")
            return

        # 오프셋 계산 (logical = raw + offset) -> offset = 2048 - raw
        offset = LOGICAL_CENTER - raw_pos

        check_logical = raw_pos + offset

        # 결과 출력
        print(
            f"[결과] ID {target_id} -> "
            f"raw={raw_pos:4d}, "
            f"offset={offset:6d}, "
            f"raw+offset={check_logical:4d}"
        )
        print("-" * 70)
        
        print("\n완료되었습니다.")
        print(f"출력된 offset 값({offset})을 메모해 두거나 소프트웨어 제어 코드에 반영하세요.")
        
    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        driver.close()


if __name__ == "__main__":
    calibrate_single_servo()