import time
import math
from motor_control import MiniFeetechDriver

def smoothstep(t):
    """
    0.0 ~ 1.0 사이의 진행률(t)을 입력받아 
    시작과 끝 속도가 부드럽게 감속되는 3차 다항식 프로파일을 반환합니다.
    """
    return t * t * (3.0 - 2.0 * t)

def move_robot_with_profile(driver, motor_ids, target_positions, duration=1.5, hz=50):
    """
    현재 위치에서 목표 위치까지 지정된 시간(duration) 동안
    Smoothstep 가감속 프로파일을 적용하여 부드럽게 동기 구동하는 안전 함수입니다.
    """
    total_steps = int(duration * hz)
    delay = 1.0 / hz

    # 1. 현재 모든 모터의 실제 물리적 위치 읽기 및 논리적 위치 역산
    current_positions = {}
    for m_id in motor_ids:
        pos = driver.get_position_filtered(m_id, samples=3)
        if pos is None:
            print(f"[⚠️ 에러] ID {m_id} 모터 피드백 실패. 안전을 위해 시퀀스를 정지합니다.")
            return False
        
        # 드라이버의 offset 매커니즘 복원 (물리 위치 + 오프셋 = 논리 위치)
        offset = driver.offsets.get(m_id, 0)
        logical_current = (pos + offset) % 4096
        current_positions[m_id] = logical_current

    # 2. 가감속 루프 제어
    for step in range(1, total_steps + 1):
        t = step / total_steps          # 선형 진행률 (0.0 ~ 1.0)
        s_t = smoothstep(t)             # 가감속이 적용된 진행률 (0.0 ~ 1.0)

        for m_id in motor_ids:
            cur = current_positions[m_id]
            tgt = target_positions[m_id]

            # 가감속 궤적 보간 연산
            interpolated_logical = cur + (tgt - cur) * s_t
            
            # 오프셋 드라이버 함수 호출하여 모터 구동
            driver.set_offset_position(m_id, int(interpolated_logical))
        
        time.sleep(delay)
    
    return True

if __name__ == "__main__":
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        print("[1] Feetech 모터 드라이버 오프셋 매핑 데이터 로드 중...")
        driver.load_all_offsets(MOTOR_IDS)
        time.sleep(0.5)

        # 안전을 위해 토크 상태 재강제
        for m_id in MOTOR_IDS:
            driver.set_torque(m_id, True)
        print("[*] 모든 관절 모터 토크 잠금 완료 (Holding 상태)")
        time.sleep(0.5)

        # -----------------------------------------------------------------
        # 📍 하드웨어 검증용 Pick and Place 핵심 포즈 정의 (Logical Tick)
        # 2048 기준으로 정렬된 상태에서 안전 범위(±400틱 내외)로 가동하도록 설계
        # 6번 관절(그리퍼)은 현재 더미 상태이므로 가상으로 열고 닫는 각도만 주입
        # -----------------------------------------------------------------
        pose_home      = {1: 2048, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2048} # 대기 자세
        
        pose_pick_ready= {1: 2348, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 2448} # A 상공 (그리퍼 열림 모사)
        pose_pick_down = {1: 2348, 2: 2248, 3: 1848, 4: 2248, 5: 2048, 6: 2448} # A 하강 (집기 직전)
        pose_pick_clasp= {1: 2348, 2: 2248, 3: 1848, 4: 2248, 5: 2048, 6: 1748} # A 하강 (그리퍼 닫힘 모사)
        
        pose_place_ready={1: 1748, 2: 2048, 3: 2048, 4: 2048, 5: 2048, 6: 1748} # B 상공 이동 (들고 이동)
        pose_place_down= {1: 1748, 2: 2248, 3: 1848, 4: 2248, 5: 2048, 6: 1748} # B 하강 (내려놓기)
        pose_place_open= {1: 1748, 2: 2248, 3: 1848, 4: 2248, 5: 2048, 6: 2448} # B 하강 (그리퍼 열기 모사)

        # 시스템 시작 전 Home 포즈로 부드럽게 자동 정렬 (안전 확보)
        print("\n[*] 초기 정렬: 로봇을 홈 자세(2048 일직선)로 서서히 이동합니다...")
        move_robot_with_profile(driver, MOTOR_IDS, pose_home, duration=2.0)
        time.sleep(1.5)

        # -----------------------------------------------------------------
        # 🔄 Pick and Place 5회 반복 메인 루프
        # -----------------------------------------------------------------
        REPEAT_COUNT = 5
        print(f"\n🚀 실제 로봇 Pick and Place 테스트를 시작합니다. (총 {REPEAT_COUNT}회 반복)")
        
        for loop in range(1, REPEAT_COUNT + 1):
            print(f"\n==================== [LOOP {loop} / {REPEAT_COUNT}] ====================")

            # Step 1: 물체 정점 A 상공으로 이동 + 그리퍼 개방
            print("[Step 1] 물체 집기 위치(A) 상공으로 이동 중...")
            move_robot_with_profile(driver, MOTOR_IDS, pose_pick_ready, duration=1.5)
            time.sleep(0.3)

            # Step 2: 물체를 잡기 위해 아래로 하강
            print("[Step 2] 집기 위치(A)로 하강 중...")
            move_robot_with_profile(driver, MOTOR_IDS, pose_pick_down, duration=1.2)
            time.sleep(0.3)

            # Step 3: 물체 파지 (그리퍼 닫힘 명령 발생)
            print("[Step 3] [더미 동작] 그리퍼로 물체를 파지합니다.")
            move_robot_with_profile(driver, MOTOR_IDS, pose_pick_clasp, duration=0.8)
            time.sleep(0.5) # 잡는 순간 하드웨어 안정화를 위한 대기

            # Step 4: 물체를 들어 올려 목표지 B 상공으로 안전 이동
            print("[Step 4] 물체를 들고 목표 위치(B) 상공으로 탈출 및 이동 중...")
            move_robot_with_profile(driver, MOTOR_IDS, pose_place_ready, duration=1.8)
            time.sleep(0.3)

            # Step 5: 목표지 B에 내려놓기 위해 하강
            print("[Step 5] 목표 위치(B) 지면으로 하강 중...")
            move_robot_with_profile(driver, MOTOR_IDS, pose_place_down, duration=1.2)
            time.sleep(0.3)

            # Step 6: 물체 해제 (그리퍼 열림 명령 발생)
            print("[Step 6] [더미 동작] 그리퍼를 열어 물체를 놓아줍니다.")
            move_robot_with_profile(driver, MOTOR_IDS, pose_place_open, duration=0.8)
            time.sleep(0.5)

        # 모든 루프 종료 후 로봇을 다시 가장 안전한 홈 포즈로 귀환시킵니다.
        print("\n==================================================")
        print("[*] 5회 작업 완료! 안전 정지 상태(Home 자세)로 귀환합니다...")
        move_robot_with_profile(driver, MOTOR_IDS, pose_home, duration=2.0)
        print("[🎉] 모든 시퀀스가 완벽하게 종료되었습니다.")

    except KeyboardInterrupt:
        print("\n[⚠️ 비상 중단] 사용자가 Ctrl+C를 눌러 작업을 강제 중단했습니다.")
        
    finally:
        # 안전하게 시리얼 버스를 닫아 자원을 해제합니다.
        driver.close()