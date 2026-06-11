import time
from motor_control import MiniFeetechDriver

if __name__ == "__main__":
    # --- 설정 영역 ---
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    
    TARGET_MOTOR_ID = None  # <<-- 테스트하고 싶은 모터의 ID를 여기 적어주세요 
    TARGET_POS = 2048    # STS3215의 물리적 중심점 [cite: 1, 4]
    ALLOWABLE_ERROR = 10 # 도달 인정 오차 범위 (±10)

    print("서보머터 중심이동하기")
    driver = MiniFeetechDriver(PORT, BAUDRATE)
    time.sleep(1)
    TARGET_MOTOR_ID = int(input("중심으로 이동할 서보를 선택: "))

    try:
        print(f"\n[*] ID {TARGET_MOTOR_ID}번 모터 설정 시작...")
        
        # 1. 해당 모터의 토크(힘)를 켭니다. [cite: 3]
        driver.set_torque(TARGET_MOTOR_ID, True)
        time.sleep(0.05)
        
        # 2. 물리적 절대 원점인 2048 위치로 이동 명령을 보냅니다. [cite: 4]
        driver.set_position(TARGET_MOTOR_ID, TARGET_POS)
        print(f" -> ID {TARGET_MOTOR_ID}: 토크 ON 및 목표 위치({TARGET_POS}) 전송 완료 [cite: 3, 4]")

        print(f"\n[*] ID {TARGET_MOTOR_ID}번 모터 이동 모니터링 시작...")
        
        while True:
            # 현재 위치 읽기 (필터링 적용) [cite: 5]
            current_pos = driver.get_position_filtered(TARGET_MOTOR_ID, samples=3)
            
            if current_pos is None:
                print(f"[경고] ID {TARGET_MOTOR_ID}: 데이터를 읽을 수 없습니다. 연결을 확인하세요.", end="\r")
            else:
                # 목표치와의 오차 계산
                error = abs(current_pos - TARGET_POS)
                print(f"현재 위치: {current_pos:4d} (목표: {TARGET_POS}, 오차: {error})      ", end="\r", flush=True)
                
                # 지정한 오차 범위 내로 들어오면 루프 종료 [cite: 5]
                if error <= ALLOWABLE_ERROR:
                    print(f"\n\n[★] ID {TARGET_MOTOR_ID}번 모터가 중심점({TARGET_POS})에 도달했습니다! [cite: 2, 4]")
                    break
            
            time.sleep(0.1)

        # 토크를 유지하여 조립 중에 서보가 돌아가지 않도록 붙잡아둡니다. [cite: 8, 9]
        print(f"[!] ID {TARGET_MOTOR_ID}번 모터가 고정되었습니다. 이 상태에서 조립을 진행하세요. [cite: 8, 9]")
        print("[*] 종료하려면 엔터(Enter) 키를 누르거나 Ctrl+C를 누르세요. [cite: 11]")
        input()

    except KeyboardInterrupt:
        print("\n\n[-] 사용자에 의해 중단되었습니다. [cite: 11]")

    finally:
        # 시리얼 포트 자원 해제 [cite: 11]
        driver.close()