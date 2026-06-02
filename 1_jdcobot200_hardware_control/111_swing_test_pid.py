import time
import math
from motor_control import MiniFeetechDriver


if __name__ == "__main__":
    PORT = "COM12"
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    DEG_TO_TICK = 4096.0 / 360.0
    SWING_DEG = 10 
    SWING_TICK = int(SWING_DEG * DEG_TO_TICK)
    
    # --- [성능 최적화 및 진동 저감 설정] ---
    # 모터 내부의 하드웨어 가감속(Profile) 설정을 지원하는지 확인 후 세팅합니다.
    # 지원하지 않는 기본 드라이버일 경우를 대비해 아래 try-except 처리를 해두었습니다.
    try:
        for motor_id in MOTOR_IDS:
            # Feetech 모터의 Acceleration 레지스터를 활성화하여 자체 가감속 유도 (진동 방지 핵심)
            # 드라이버에 관련 메서드가 있다면 활용하고, 없다면 패스합니다.
            if hasattr(driver, 'set_acceleration'):
                driver.set_acceleration(motor_id, 50)  # 가속도 값 설정 (0~254)
            if hasattr(driver, 'set_speed'):
                driver.set_speed(motor_id, 1000)       # 목표 최고 속도 향상
    except Exception as e:
        print(f"하드웨어 가감속 레지스터 설정 건너뜀: {e}")

    # 부드러운 사인파(Sine) 궤적 제어를 위한 파라미터
    TOTAL_STEPS = 80      # 하나의 모션(예: 중심->+10도)을 몇 단계로 쪼갤 것인가
    STEP_DELAY = 0.015    # 루프 주기 단축 (기존 0.03초 -> 0.015초로 속도 향상 및 제어 정밀도 업)
    HOLD_TIME = 0.2       # 관절 전환 시 대기 시간 줄임

    try:
        print("[1] 현재 위치를 기준 위치로 읽습니다.")
        center_positions = {}

        for motor_id in MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=7) # 샘플링 수를 늘려 노이즈 차단

            if pos is None:
                print(f"ID {motor_id}: 위치 읽기 실패")
                driver.close()
                exit()

            center_positions[motor_id] = pos
            driver.set_torque(motor_id, True)
            time.sleep(0.05)

        print("기준 위치:", center_positions)
        print("[2] 사인파 가감속을 통해 진동 없이 부드럽고 빠르게 순차 구동을 시작합니다.")
        print("Ctrl+C로 종료")

       
        for motor_id in MOTOR_IDS:
            print(f"-> 현재 구동 중인 모터 ID: {motor_id}")
            center = center_positions[motor_id]

            # 1. 중심(0) -> +10도 (사인파 가속)
            for i in range(TOTAL_STEPS + 1):
                # 0에서 pi/2 까지 변화 공식 이용
                ratio = math.sin((i / TOTAL_STEPS) * (math.pi / 2))
                target = center + (SWING_TICK * ratio)
                target = max(0, min(4095, int(target)))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # 2. +10도 -> -10도 (최고속도를 거쳐 다시 감속하는 완전한 정현파 가감속)
            for i in range(TOTAL_STEPS * 2 + 1):
                # pi/2에서 3*pi/2 까지 변화
                ratio = math.sin((math.pi / 2) + (i / (TOTAL_STEPS * 2)) * math.pi)
                target = center + (SWING_TICK * ratio)
                target = max(0, min(4095, int(target)))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

            # 3. -10도 -> 다시 중심(0) 복귀 (사인파 감속)
            for i in range(TOTAL_STEPS + 1):
                # 3*pi/2에서 2*pi 까지 변화
                ratio = math.sin((3 * math.pi / 2) + (i / TOTAL_STEPS) * (math.pi / 2))
                target = center + (SWING_TICK * ratio)
                target = max(0, min(4095, int(target)))
                driver.set_position(motor_id, target)
                time.sleep(STEP_DELAY)

            time.sleep(HOLD_TIME)

    except KeyboardInterrupt:
        print("\n종료합니다.")

    finally:
        # 종료 시 토크를 안전하게 해제하고 서보를 보호하려면 아래 주석을 해제하세요.
        # for motor_id in MOTOR_IDS:
        #     driver.set_torque(motor_id, False)
        driver.close()