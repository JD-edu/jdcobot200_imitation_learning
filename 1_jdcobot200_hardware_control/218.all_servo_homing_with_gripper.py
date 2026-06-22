import serial
import time
import math
from motor_control import MiniFeetechDriver  # 환경에 맞게 파일명/클래스명을 조정하세요.

def smoothstep(t):
    """
    t가 0에서 1까지 변할 때, 0에서 1까지 부드러운 S-곡선 가속/감속 값을 반환 (3t^2 - 2t^3)
    """
    return t * t * (3 - 2 * t)

def load_offsets_from_file(file_path="offsets.txt"):
    """
    통합 조인트 오프셋 파일(offsets.txt)에서 1~6번(그리퍼 포함) 모터 오프셋 데이터를 로드합니다.
    """
    offsets = {}
    
    # 1~6번 관절 및 그리퍼 오프셋 통합 파일 로드
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    motor_id, offset_val = line.split("=")
                    offsets[int(motor_id)] = int(offset_val)
        print(f"▶ [{file_path}] 통합 오프셋 로드 완료 (총 모터 개수: {len(offsets)}개)")
    except FileNotFoundError:
        print(f"[경고] {file_path} 파일이 존재하지 않습니다. 모든 관절 오프셋을 0으로 초기화합니다.")
        for i in range(1, 7):
            offsets[i] = 0
        
    print(f"✨ 최종 통합 오프셋 맵 구성 완료: {offsets}\n")
    return offsets

if __name__ == "__main__":
    # --- 통신 및 프로파일 설정 ---
    PORT = "/dev/ttyACM0" 
    BAUDRATE = 1000000 
    THEORETICAL_CENTER = 2048  # STS3215의 이론상 물리적 중심점 
    
    # 가감속 프로파일 시간 설정 (그리퍼 결합으로 부하 증가 대비 유연한 속도 세팅)
    DURATION = 3.5             # 홈잉 이동 총 소요 시간 (초 단위)
    CONTROL_PERIOD = 0.02      # 제어 주기 (20ms = 0.02초, 통신 최적 주기)
    TOTAL_STEPS = int(DURATION / CONTROL_PERIOD)

    print("==================================================================")
    print("  ★ [통합 마스터] 1~5번 관절 + 6번 그리퍼 프로파일 가감속 홈잉 ★  ")
    print("==================================================================")

    # 1. 단일 offsets.txt 파일로부터 1~6번 오프셋 딕셔너리 확보
    all_motor_offsets = load_offsets_from_file("offsets.txt")

    # --- 드라이버 초기화 및 통신 연결 ---
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE) 
    
    # 2. 모든 타겟 모터 안전 점검 및 시작/목표 위치 계산
    start_positions = {}
    target_home_positions = {}
    
    print("[단계 1] 전 관절 및 6번 그리퍼 개별 토크 ON 및 실시간 현재 위치 측정")
    print("-" * 66)
    
    for motor_id, offset in sorted(all_motor_offsets.items()):
        # 안전을 위해 모든 서보 및 그리퍼의 토크(힘)를 순차적으로 켭니다.
        driver.set_torque(motor_id, True)
        time.sleep(0.04)
        
        # 주입된 정밀 모터 피드백 함수 호출
        current_pos = driver.get_position(motor_id)
        
        if current_pos is None or current_pos == 0: 
            print(f"[ID: {motor_id:02d}] 위치 피드백을 읽지 못했습니다. 통신선 결선 상태를 체크하세요.")
            continue
            
        start_positions[motor_id] = current_pos
        # 소프트웨어 기준 원점 계산 방식 적용 (2048 + 각각의 파일별 고유 오프셋)
        target_home_positions[motor_id] = THEORETICAL_CENTER + offset
        
        name_tag = "그리퍼 (6번)" if motor_id == 6 else f"관절 {motor_id}번"
        print(f" -> {name_tag:11s} 현재 위치: {current_pos:4d} | 오프셋: {offset:+3d} | 최종 소프트웨어 원점: {target_home_positions[motor_id]:4d}")

    if not start_positions:
        print("정상 연동된 서보모터가 한 개도 발견되지 않았습니다. 종료합니다.")
        exit()

    print("\n------------------------------------------------------------------")
    print(f"🚀 동시 가속/감속 다중 축 타임 슬라이싱 제어 구동 시작! ({DURATION}초간 진행)") 
    print("------------------------------------------------------------------")

    # 3. 통합 다중 축 시간 분할(Time-Splitting) S-curve 가감속 제어 루프
    try:
        for step in range(TOTAL_STEPS + 1):
            t = step / TOTAL_STEPS  # 0.0 ~ 1.0 진행률
            alpha = smoothstep(t)   # S-곡선 가중치 매핑
            
            # 모든 관절과 그리퍼가 미세 지연을 최소화하며 버스 통신으로 나누어 진입
            for motor_id in list(start_positions.keys()):
                start_p = start_positions[motor_id]
                target_p = target_home_positions[motor_id]
                
                # 정밀 타겟 분할 위치 보간 연산
                interpolated_pos = int(start_p + (target_p - start_p) * alpha)
                
                # 궤적 동시 주입
                driver.set_position(motor_id, interpolated_pos)
                time.sleep(0.0015)  # 명령 패킷 연속 충돌 방지를 위한 초미세 가이드 지연
            
            # 20ms 정밀 통신 주기 동기화
            time.sleep(CONTROL_PERIOD)
            
        print("\n🎉 [성공] 모든 로봇 조인트 및 그리퍼가 충격 없이 부드럽게 원점에 안착했습니다!")

        # 4. 최종 수렴 상태 정밀 도달 체크 모니터링
        print("\n[단계 2] 전 축 최종 수렴 정렬 상태 모니터링 (종료하려면 Ctrl+C)...")
        print("-" * 66)
        
        while True:
            all_reached = True
            for motor_id, target_p in target_home_positions.items():
                curr_p = driver.get_position(motor_id)
                if curr_p is not None:
                    error = abs(curr_p - target_p)
                    name_tag = "그리퍼" if motor_id == 6 else f"관절 {motor_id}"
                    print(f"[{name_tag:5s}] 현재 위치: {curr_p:4d} / 목표 원점: {target_p:4d} (잔여 오차: {error:2d})")
                    
                    # 허용 정밀 오차 범위 (+-5 스텝) 확인
                    if error > 5:
                        all_reached = False
                else:
                    all_reached = False
            
            if all_reached:
                print("\n완벽 정렬 완료! 전 조인트 및 그리퍼 기구부가 소프트웨어 영점에 락(Freeze)되었습니다.")
                break
                
            print("..정밀 안착 조율 중.. (완전 종료는 Ctrl+C를 누르세요)\n")
            time.sleep(1.0)

        print("-" * 66)
        print("현재 모든 프레임에 강력한 고정 토크가 유지되고 있어 안전합니다.")
        print("작업을 마치고 마스터 프로그램을 안전하게 종료하려면 [Enter]를 누르세요.")
        input()
        print("▶ 시스템 운용을 안전하게 마칩니다.")

    except KeyboardInterrupt:
        print("\n▶ [안내] 모니터링을 종료하거나 제어를 중단합니다. 현재 위치에서 전 서보 안전 잠금(Torque ON)을 유지합니다.")