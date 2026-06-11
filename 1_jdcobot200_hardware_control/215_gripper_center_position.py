import time
import serial
from motor_control import MiniFeetechDriver  # 사용자 환경에 맞는 파일명/클래스명

def move_gripper_to_pure_2048():
    # ==========================================
    # [설정 영역] 그리퍼 ID 및 포트 설정
    # ==========================================
    PORT = "/dev/ttyACM0"       # 시리얼 포트 경로 (환경에 따라 /dev/ttyUSB0 등으로 변경)
    BAUDRATE = 1000000         # 통신 속도 (1Mbps)
    GRIPPER_MOTOR_ID = 6       # 그리퍼 서보모터 ID
    PURE_TARGET_POS = 2048     # 이동할 물리적 정중앙 위치

    # 1. 드라이버 초기화 및 연결
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    
    print(f"▶ [ID: {GRIPPER_MOTOR_ID}] 그리퍼를 오프셋 없는 순수 2048 위치로 이동합니다.")
    
    try:
        # 2. 안전을 위해 그리퍼 모터의 토크(힘)를 먼저 켭니다.
        driver.set_torque(GRIPPER_MOTOR_ID, True)
        time.sleep(0.1)
        
        # 3. 다른 모터는 건드리지 않고, 그리퍼에만 다이렉트로 2048 이동 명령 전송 
        driver.set_position(GRIPPER_MOTOR_ID, PURE_TARGET_POS)
        print(f"그리퍼가 {PURE_TARGET_POS} 위치로 이동 중입니다...")
        
        # 4. 목표 위치에 도달할 때까지 실시간 모니터링 대기
        while True:
            # 현재 위치 레지스터(56)에서 2바이트 데이터를 읽어옴 [cite: 225]
            current_pos = driver.read_u16(GRIPPER_MOTOR_ID, 56)
            print(f"\r   현재 위치: {current_pos} / 목표: {PURE_TARGET_POS}", end="", flush=True)
            
            # 목표치 근처(오차범위 ±5 스텝 이내)에 들어오면 루프 탈출
            if abs(current_pos - PURE_TARGET_POS) <= 5:
                print(f"\n\n✅ [ID: {GRIPPER_MOTOR_ID}] 그리퍼가 순수 2048 지점에 안착했습니다.")
                break
            time.sleep(0.1)
            
        # 5. 토크가 단단히 걸린 상태(Freeze)를 유지하며 대기
        input("\n[안내] 정렬이 완료되었습니다. 프로그램을 안전하게 종료하려면 Enter를 누르세요...")
        print("▶ 프로그램을 종료합니다. (그리퍼 고정 상태 유지)")

    except KeyboardInterrupt:
        print("\n▶ [안내] 사용자에 의해 프로그램이 정지되었습니다. 안전을 위해 토크 고정은 유지됩니다.")

if __name__ == "__main__":
    move_gripper_to_pure_2048()