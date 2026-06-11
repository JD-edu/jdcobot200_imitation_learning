import time
import os
from motor_control import MiniFeetechDriver

def calibrate_all_servos():
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    LOGICAL_CENTER = 2048
    
    # 교정할 로봇암의 전체 서보 ID 리스트 (환경에 맞게 수정 가능)
    MOTOR_IDS = [1, 2, 3, 4, 5]
    OUTPUT_FILE = "offsets.txt"

    driver = MiniFeetechDriver(PORT, BAUDRATE)
    
    # 각 모터의 최종 오프셋을 저장할 딕셔너리
    calibrated_offsets = {}

    try:
        print("=" * 70)
        print("STS3215 All Motors Homing Offset Calibration & Save")
        print("=" * 70)
        print(f"대상 서보 ID 리스트: {MOTOR_IDS}")
        print("각 모터를 차례대로 교정합니다. 준비되면 Enter를 누르세요.")
        input("시작하려면 [Enter]를 누르세요... ")

        for target_id in MOTOR_IDS:
            print("\n" + "=" * 50)
            print(f"▶ [ID {target_id}번 모터 교정 시작]")
            print("=" * 50)

            # 1. 대상 모터의 토크만 해제 (다른 모터는 원래 상태 유지하여 로봇암 지탱)
            print(f"[1] ID {target_id}번 모터 토크 OFF")
            driver.set_torque(target_id, False) 
            time.sleep(0.05)

            # 2. 사용자 수동 정렬 대기
            print(f"→ ID {target_id}번 관절을 손으로 원하는 중심 자세(원점)에 맞추세요.")
            input("   자세를 맞춘 후 [Enter]를 누르세요... ")

            # 3. 현재 엔코더 위치 읽기 (7번 샘플링 필터링 적용)
            print(f"[2] ID {target_id}번 현재 엔코더 값 읽는 중...")
            raw_pos = driver.get_position_filtered(target_id, samples=7)

            if raw_pos is None:
                print(f"ID {target_id}: 엔코더 값을 읽는데 실패했습니다. (기본값 0으로 처리)")
                offset = 0
            else:
                # 오프셋 계산 (원점 2048 - 현재 raw 값)
                offset = LOGICAL_CENTER - raw_pos 
                check_logical = raw_pos + offset

                # 측정 결과 실시간 출력
                print("-" * 50)
                print(
                    f"   [측정 결과] ID {target_id} -> "
                    f"raw={raw_pos:4d}, "
                    f"offset={offset:6d}, "
                    f"raw+offset={check_logical:4d}"
                )
                print("-" * 50)

            # 결과를 딕셔너리에 저장
            calibrated_offsets[target_id] = offset

            # 4. 교정이 끝난 모터는 다시 토크를 켜서 현재 수동으로 맞춘 위치를 고정
            print(f"[3] ID {target_id}번 모터 현재 위치로 토크 ON (자세 유지)")
            if raw_pos is not None:
                driver.set_position(target_id, raw_pos) # 현재 위치로 명령 전달 후 토크ON 효과
            driver.set_torque(target_id, True)
            time.sleep(0.05)

        # 모든 모터 교정 완료 후 파일 저장 단계
        print("\n" + "=" * 70)
        print("모든 모터 교정 완료! 파일 저장을 시작합니다.")
        print("=" * 70)

        # 직관적인 ID=오프셋 포맷으로 텍스트 파일 쓰기
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for m_id in MOTOR_IDS:
                f.write(f"{m_id}={-1*calibrated_offsets[m_id]}\n")

        print(f"오프셋 저장 완료: '{os.path.abspath(OUTPUT_FILE)}'")
        print("저장된 파일 내용 미리보기:")
        for m_id in MOTOR_IDS:
            print(f"   ID {m_id} = {calibrated_offsets[m_id]}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 칼리브레이션이 중단되었습니다. 파일이 저장되지 않았을 수 있습니다.")

    finally:
        # 종료 시 모든 모터의 통신 연결 안전하게 닫기
        driver.close()
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    calibrate_all_servos()