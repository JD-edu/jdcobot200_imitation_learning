import serial
import time
from motor_control import MiniFeetechDriver  # 환경에 맞게 파일명/클래스명을 조정하세요. 

if __name__ == "__main__":
    # --- 통신 설정 ---
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    THEORETICAL_CENTER = 2048  # STS3215의 이론상 물리적 중심점 [cite: 23]
    print("단일 서보 홈잉 프로그램")
   
    # 1. 사용자로부터 제어할 서보 ID와 오프셋 값을 입력받습니다. 
    try:
        TARGET_MOTOR_ID = int(input("▶ 제어할 서보 모터 ID를 입력하세요: "))
        SOFTWARE_OFFSET = int(input("▶ 적용할 소프트웨어 오프셋 값을 입력하세요 (예: 62 또는 -20): "))
    except ValueError:
        print("올바른 숫자를 입력해야 합니다. 프로그램을 종료합니다.")
        exit()

    # 최종적으로 도달해야 하는 소프트웨어 기준 원점 계산 [cite: 3]
    TARGET_HOME_POSITION = THEORETICAL_CENTER + SOFTWARE_OFFSET

    # --- 드라이버 초기화 및 통신 연결 ---
    # MiniFeetechDriver 내부 구현 구조에 맞게 초기화 형식을 맞춰주세요. 
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    
    print("\n--------------------------------------------------")
    print(f"[{TARGET_MOTOR_ID}번 모터] 소프트웨어 홈잉 프로세스를 시작합니다. [cite: 3]")
    print(f"이론상 중심: {THEORETICAL_CENTER} | 입력 오프셋: {SOFTWARE_OFFSET} | 최종 목표 원점: {TARGET_HOME_POSITION} [cite: 3]")
    print("--------------------------------------------------")

    try:
        # 1. 안전을 위해 해당 모터의 토크(힘)를 켭니다. [cite: 9, 38]
        driver.set_torque(TARGET_MOTOR_ID, True)
        time.sleep(0.1)
        
        # 2. 서보가 현재 어떤 위치에 있든 상관없이, 오프셋이 반영된 원점 위치로 이동 명령을 내립니다. [cite: 3]
        print(f"-> {TARGET_MOTOR_ID}번 모터를 원점({TARGET_HOME_POSITION})으로 이동시킵니다... [cite: 3]")
        driver.set_position(TARGET_MOTOR_ID, TARGET_HOME_POSITION)
        
        # 3. 목표 위치에 잘 도달했는지 현재 위치를 모니터링합니다. [cite: 11]
        while True:
            # 2바이트 위치 레지스터(REG_PRESENT_POSITION=56)에서 현재 위치 읽기 [cite: 33]
            current_pos = driver.read_u16(TARGET_MOTOR_ID, 56)
            
            if current_pos is not None:
                print(f"현재 위치: {current_pos} / 목표 원점: {TARGET_HOME_POSITION}")
                
                # 목표 위치 오차 범위(+-5 스텝) 안으로 들어오면 정렬 완료로 판단
                if abs(current_pos - TARGET_HOME_POSITION) <= 5:
                    print(f"\n[성공] {TARGET_MOTOR_ID}번 서보 모터가 정확하게 오프셋 원점에 도달하여 고정되었습니다!")
                    break
            else:
                print("모터 위치를 읽어오는데 실패했습니다. 통신 상태를 확인하세요.")
                
            time.sleep(0.2)
            
        # 4. 정렬된 상태를 유지하며 사용자 대기 (조립 확인 또는 다음 작업을 위한 대기) [cite: 14]
        print("--------------------------------------------------")
        print("💡 현재 원점 포지션으로 힘(토크)이 들어간 채 고정되어 있습니다. [cite: 14, 15]")
        print("프로그램을 안전하게 종료하려면 [Enter] 키를 누르세요. (토크가 유지됩니다) [cite: 14, 17]")
        input()

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다. [cite: 17]")