import time
import os
import serial
from motor_control import MiniFeetechDriver  # 사용자 환경에 맞는 파일명/클래스명

def calibrate_gripper_and_save():
    # ==========================================
    # [설정 영역] 그리퍼 ID, 포트 및 저장 파일 설정
    # ==========================================
    PORT = "/dev/ttyACM0"           # 시리얼 포트 경로 (환경에 따라 변경) [cite: 130]
    BAUDRATE = 1000000             # 통신 속도 (1Mbps)
    GRIPPER_MOTOR_ID = 6           # 그리퍼 서보모터 ID [cite: 57]
    THEORETICAL_CENTER = 2048      # STS3215의 이론상 정중앙 기준점 [cite: 50]
    SAVE_FILE_NAME = "offsets.txt" # 통합 관리할 조인트 오프셋 파일명 [cite: 59]

    # 1. 드라이버 초기화
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    
    print("=" * 60)
    print(f"★ [ID: {GRIPPER_MOTOR_ID}] 그리퍼 소프트웨어 칼리브레이션 (`offsets.txt` 통합 버전) ★")
    print("=" * 60)
    
    try:
        # 2. 그리퍼 모터만 콕 집어서 토크 해제 (Release) [cite: 33, 59]
        driver.set_torque(GRIPPER_MOTOR_ID, False)
        print(f"\n🔓 그리퍼 토크 해제 완료! [다른 1~5번 관절은 안전하게 고정 상태 유지]") 
        print("💡 [작업] 양쪽 그리퍼 이빨이 바닥과 완벽히 수직/평행이 되도록 손으로 맞추세요.") 
        
        # 사용자가 정렬을 완료할 때까지 대기 [cite: 33]
        input("👉 정렬을 완료했다면 [Enter] 키를 누르세요...")
        
        # 3. 정렬된 상태의 실제 물리 엔코더 값 읽기 (레지스터 56번) [cite: 200, 295]
        current_pos = driver.read_u16(GRIPPER_MOTOR_ID, 56) 
        
        # 4. 오프셋 계산 (실제 위치 - 이론상 중심점) [cite: 7, 72]
        gripper_offset = current_pos - THEORETICAL_CENTER
        
        print("-" * 60)
        print(f"   측정된 현재 엔코더 값: {current_pos}")
        print(f"   이론상 정중앙 기준점: {THEORETICAL_CENTER}")
        print(f"   ✨ 계산된 그리퍼 오프셋: {gripper_offset}")
        print("-" * 60)
        
        # 5. 기존 offsets.txt 파일 내용을 읽어와 6번 데이터만 갱신하기 
        existing_offsets = {}
        if os.path.exists(SAVE_FILE_NAME):
            with open(SAVE_FILE_NAME, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        motor_id, offset_val = line.split("=")
                        existing_offsets[int(motor_id)] = int(offset_val) 
        
        # 6번(그리퍼) 오프셋 데이터 추가 또는 업데이트
        existing_offsets[GRIPPER_MOTOR_ID] = gripper_offset
        
        # 6. 정렬된 데이터를 정렬(ID 순)하여 offsets.txt에 최종 저장
        with open(SAVE_FILE_NAME, "w", encoding="utf-8") as f:
            f.write("# jdcobot200 Servo Motor Calibration Offsets\n")
            for m_id in sorted(existing_offsets.keys()):
                f.write(f"{m_id}={existing_offsets[m_id]}\n")
            
        print(f"💾 [{SAVE_FILE_NAME}] 파일에 6번 그리퍼 오프셋이 성공적으로 통합 저장되었습니다! ") 
        
        # 7. 작업이 끝났으므로 그리퍼 서보를 그 자리에 단단히 고정 (Freeze) [cite: 9, 33]
        driver.set_torque(GRIPPER_MOTOR_ID, True)
        print(f"🔒 그리퍼 토크가 다시 잠겼습니다. 안전하게 세팅이 종료됩니다.")
        
    except KeyboardInterrupt:
        # 도중에 Ctrl+C로 취소하더라도 안전하게 토크를 다시 걸어줌 [cite: 41, 134]
        driver.set_torque(GRIPPER_MOTOR_ID, True)
        print("\n▶ [안내] 사용자에 의해 취소되었습니다. 그리퍼 안전을 위해 토크를 다시 켭니다.")
    except Exception as e:
        driver.set_torque(GRIPPER_MOTOR_ID, True)
        print(f"\n❌ 에러 발생: {e} (안전을 위해 그리퍼 토크를 다시 켭니다.)")

if __name__ == "__main__":
    calibrate_gripper_and_save()