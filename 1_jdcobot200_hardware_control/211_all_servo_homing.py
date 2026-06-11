import serial
import time
import math
from motor_control import MiniFeetechDriver  # 환경에 맞게 파일명/클래스명을 조정하세요. [cite: 106]

def smoothstep(t):
    """
    t가 0에서 1까지 변할 때, 0에서 1까지 부드러운 S-곡선 가속/감속 값을 반환 (3t^2 - 2t^3) [cite: 126]
    """
    return t * t * (3 - 2 * t)

def load_offsets_from_file(file_path="offsets.txt"):
    """
    텍스트 파일에서 모터 ID와 오프셋 매핑 데이터를 읽어옵니다. [cite: 131, 132]
    파일 포맷 예시: 1=62 
    """
    offsets = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):  # 빈 줄이나 주석 처리된 줄 제외
                    continue
                if "=" in line:
                    motor_id, offset_val = line.split("=")
                    offsets[int(motor_id)] = int(offset_val)
        print(f"▶ [{file_path}]에서 오프셋 데이터를 성공적으로 로드했습니다: {offsets}")
    except FileNotFoundError:
        print(f"❌ 에러: {file_path} 파일이 존재하지 않습니다. 프로그램을 종료합니다.")
        return None
    except Exception as e:
        print(f"❌ 파일 읽기 중 에러 발생: {e}")
        return None
    return offsets

if __name__ == "__main__":
    # --- 통신 및 프로파일 설정 ---
    PORT = "/dev/ttyACM0" 
    BAUDRATE = 1000000 
    THEORETICAL_CENTER = 2048  # STS3215의 이론상 물리적 중심점 
    
    # 가감속 프로파일 시간 설정
    DURATION = 3.0             # 홈잉 이동 총 소요 시간 (초 단위) [cite: 127]
    CONTROL_PERIOD = 0.02      # 제어 주기 (20ms = 0.02초, 통신 최적 주기) [cite: 129, 130]
    TOTAL_STEPS = int(DURATION / CONTROL_PERIOD)

    print("==================================================")
    print("      [마스터] 전체 서보 프로파일 가감속 홈잉 프로그램      ")
    print("==================================================")

    # 1. 파일에서 오프셋 딕셔너리 가져오기
    motor_offsets = load_offsets_from_file("offsets.txt")
    if not motor_offsets:
        exit()

    # --- 드라이버 초기화 및 통신 연결 ---
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE) 
    
    # 2. 모든 타겟 모터 안전 점검 및 시작/목표 위치 계산
    start_positions = {}
    target_home_positions = {}
    
    print("\n[단계 1] 모터 초기화 및 get_position 함수를 이용한 현재 위치 측정 중...")
    for motor_id, offset in motor_offsets.items():
        # 안전을 위해 해당 모터의 토크(힘)를 켭니다. [cite: 21]
        driver.set_torque(motor_id, True)
        time.sleep(0.05)
        
        # 요청하신 driver.get_position 함수로 현재 위치 읽기
        current_pos = driver.get_position(motor_id)
        
        if current_pos is None or current_pos == 0: 
            print(f"⚠️ {motor_id}번 모터의 위치를 읽지 못했습니다. 통신 상태를 확인하세요.")
            continue
            
        start_positions[motor_id] = current_pos
        # 최종적으로 도달해야 하는 소프트웨어 기준 원점 계산 (2048 + 오프셋) [cite: 72]
        target_home_positions[motor_id] = THEORETICAL_CENTER + offset
        
        print(f" -> 모터 ID [{motor_id:02d}] 현재 위치: {current_pos:4d} | 오프셋: {offset:+3d} | 최종 목표 원점: {target_home_positions[motor_id]:4d}")

    if not start_positions:
        print("❌ 위치가 정상적으로 확인된 모터가 없습니다. 프로그램을 종료합니다.")
        exit()

    print("\n--------------------------------------------------")
    print(f"▶ 시간 분할 기반 S-Curve 가감속 홈잉을 시작합니다. ({DURATION}초 간 구동)") 
    print("--------------------------------------------------")

    # 3. 시간 분할 기반(Time-Splitting) S-curve 가감속 제어 루프 [cite: 123]
    # 모든 모터가 20ms마다 동시에 조금씩 목표 위치를 나누어 받으며 부드럽게 이동합니다. 
    try:
        for step in range(TOTAL_STEPS + 1):
            t = step / TOTAL_STEPS  # 0.0 ~ 1.0 진행률 계산 [cite: 125]
            alpha = smoothstep(t)   # S-곡선 가중치 변환 (3t^2 - 2t^3) [cite: 126]
            
            # 각 모터별로 현재 단계의 목표 마디(Interpolated)값 계산하여 전송
            for motor_id in list(start_positions.keys()):
                start_p = start_positions[motor_id]
                target_p = target_home_positions[motor_id]
                
                # 내부 보간 공식: 시작점 + (총 이동 거리 * S곡선 비율)
                interpolated_pos = int(start_p + (target_p - start_p) * alpha)
                
                # 계산된 부드러운 중간 좌표로 모터 이동 명령 전송 [cite: 129]
                driver.set_position(motor_id, interpolated_pos)
                time.sleep(0.002)
            
            # 20ms 정밀 주기 유지 [cite: 129, 130]
            time.sleep(CONTROL_PERIOD)
            
        print("\n[성공] 모든 서보 모터가 가감속 프로파일을 통해 부드럽게 원점에 안착했습니다! [cite: 111]")

        # 4. 최종 정밀 도달 상태 점검 (오차 범위 내 안착 확인)
        print("\n[단계 2] 최종 도달 위치 모니터링...")
        print("--------------------------------------------------")
        
        while True:
            all_reached = True
            for motor_id, target_p in target_home_positions.items():
                curr_p = driver.get_position(motor_id)
                if curr_p is not None:
                    error = abs(curr_p - target_p)
                    print(f"모터 [{motor_id}] 현재 위치: {curr_p} / 목표 원점: {target_p} (오차: {error})")
                    # 목표 위치 오차 범위(+-5 스텝) 안으로 들어오는지 확인
                    if error > 5:
                        all_reached = False
                else:
                    all_reached = False
            
            if all_reached:
                print("\n✨ 모든 조인트 원점 정렬 완료! 로봇암이 안전하게 고정되었습니다.")
                break
                
            time.sleep(0.5)

        print("--------------------------------------------------")
        print("💡 현재 원점 포지션으로 힘(토크)이 들어간 채 고정되어 있습니다. [cite: 26, 27]")
        print("프로그램을 안전하게 종료하려면 [Enter] 키를 누르세요. (토크가 유지됩니다) [cite: 26, 29]")
        input()

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 프로그램이 중단되었습니다. 로봇암 제어를 멈춥니다.") 