import time
from motor_control import MiniFeetechDriver

if __name__ == "__main__":
    # 제공해주신 모니터링 코드의 설정을 기반으로 매칭 (포트: ttyUSB0)
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    
    # 안전을 위해 오직 1번 베이스 모터만 제어
    BASE_MOTOR_ID = 1
    
    # 주요 레지스터 주소 정의 (드라이버 클래스 내에 없는 주소 보완)
    REG_ACCELERATION = 0x30  # 가속도 레지스터 (48번 주소)
    REG_VIOLENT_SPEED = 0x2E # 최고 속도 제한 레지스터 (46번 주소)

    # 왕복할 두 목표 위치 지정 (0 ~ 4095 범위)
    # 물리 마진을 고려하여 중앙(2048) 기준 좌우로 약 45도씩(+-512 step) 안전 반경 설정
    TARGET_A = 1536  
    TARGET_B = 2560  

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        print("[1] 1번 베이스 서보 하드웨어 프로파일 설정")
        
        # [해결] write_register 대신 클래스 내부의 write_u16 또는 _write_only 방식 활용
        # 0x03은 WRITE 명령 코드입니다. 가속도(REG_ACCELERATION) 값을 40으로 설정합니다.
        # 이 한 줄로 모터 내부 칩셋이 스스로 가감속 알고리즘(S-Curve)을 켭니다.
        driver._write_only(BASE_MOTOR_ID, 0x03, [REG_ACCELERATION, 40])
        time.sleep(0.02)
        
        # 최고 속도 제한(REG_VIOLENT_SPEED)을 150으로 설정합니다. (2바이트 데이터이므로 write_u16 활용)
        # 값이 작을수록 천천히 움직입니다.
        driver.write_u16(BASE_MOTOR_ID, REG_VIOLENT_SPEED, 150)
        time.sleep(0.02)
        
        # 토크 활성화
        driver.set_torque(BASE_MOTOR_ID, True)
        time.sleep(0.02)

        print("\n[2] 하드웨어 자율 S-Curve 왕복 구동 시작 (무한 루프)")
        print(" -> 종료하려면 터미널에서 Ctrl + C를 누르세요.")
        print("-" * 60)

        loop_count = 1
        while True:
            # ----------------------------------------------------
            # 위치 A로 이동
            # ----------------------------------------------------
            print(f"[{loop_count}회차] ➔ 목표 위치 A ({TARGET_A})로 부드럽게 이동 시작")
            driver.set_position(BASE_MOTOR_ID, TARGET_A)
            
            # 모터가 가속하고 감속하여 완전히 안착할 때까지 3초간 여유롭게 대기
            # 이 시간 동안 PC는 모터에 어떠한 패킷도 보내지 않고 휴식합니다.
            time.sleep(3.0) 
            
            # ----------------------------------------------------
            # 위치 B로 이동
            # ----------------------------------------------------
            print(f"[{loop_count}회차] ➔ 목표 위치 B ({TARGET_B})로 부드럽게 이동 시작")
            driver.set_position(BASE_MOTOR_ID, TARGET_B)
            
            time.sleep(3.0)
            
            loop_count += 1
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n[사용자 중단] 테스트를 종료하고 안전하게 통신을 닫습니다.")
    finally:
        driver.close()