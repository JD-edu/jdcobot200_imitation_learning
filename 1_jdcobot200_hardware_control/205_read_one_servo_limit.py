import serial
import time
from motor_control import MiniFeetechDriver  # 환경에 맞게 파일명을 수정하세요

# ==========================================
# [설정 영역] 테스트할 모터 ID와 포트 설정
# ==========================================
TARGET_MOTOR_ID = None       # 가동 범위를 측정할 모터 ID (원하는 ID로 변경 가능)
PORT = "/dev/ttyACM0"      # 시리얼 포트 경로
BAUDRATE = 1000000         # 통신 속도 (1Mbps)
# ==========================================

def monitor_joint_limit():
    # 1. 드라이버 초기화 및 통신 연결
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    # 만약 라이브러리에 시리얼 연결 메서드가 있다면 호출 (예: driver.connect(PORT, BAUDRATE))
    
    print("서보모터의 가동범위를 측정")
    TARGET_MOTOR_ID = int(input("리미트를 측정할 서보의 번호를 입력: "))
    
    try:
        # 2. 타겟 모터만 토크 해제 (손으로 움직일 수 있는 상태로 전환)
        # MiniFeetechDriver의 set_torque 함수를 활용하여 False(OFF) 전달
        driver.set_torque(TARGET_MOTOR_ID, False)
        print(f"ID {TARGET_MOTOR_ID}번 모터의 토크가 해제되었습니다.")
        print("지금 로봇암을 손으로 천천히 잡고 '시계 방향 끝'과 '반시계 방향 끝'으로 움직여보세요.")
        print("실시간으로 최소값과 최대값이 갱신되며 화면에 표시됩니다.")
        print("--------------------------------------------------")
        
        # 가동 범위 측정을 위한 변수 초기화 (STS3215의 이론상 범위는 0 ~ 4095)
        min_pos = 4095
        max_pos = 0
        
        # 3. 실시간 모니터링 루프
        while True:
            # REG_PRESENT_POSITION(56) 주소에서 2바이트 현재 위치 읽기
            # read_word가 없는 구조이므로, 제공된 read_u16 메서드를 사용합니다.
            current_pos = driver.read_u16(TARGET_MOTOR_ID, driver.REG_PRESENT_POSITION)
            
            # 통신 오류 등으로 인해 정상적인 범위를 벗어난 값이 들어오면 예외 처리
            if current_pos is None or current_pos < 0 or current_pos > 4095:
                time.sleep(0.01)
                continue
                
            # 최소값 및 최대값 실시간 갱신 및 비교
            is_updated = False
            if current_pos < min_pos:
                min_pos = current_pos
                is_updated = True
            if current_pos > max_pos:
                max_pos = current_pos
                is_updated = True
                
            # 값이 갱신될 때마다 터미널에 깔끔하게 출력 (\r을 사용하여 한 줄에서 실시간 업데이트)
            print(f" [실시간] 현재 위치: {current_pos:4d}  |  ★ 측정된 가동 범위: {min_pos:4d} ~ {max_pos:4d}", end="\r")
            
            time.sleep(0.02)  # 50Hz 주기로 부드럽게 센싱 및 출력
            
    except KeyboardInterrupt:
        # 사용자가 Ctrl+C를 눌러 프로그램을 종료했을 때 안전하게 빠져나옴
        print("\n--------------------------------------------------")
        print(f"최종 기록된 [ID {TARGET_MOTOR_ID}] 안전 가동 범위")
        print(f"   최소 제한값 (MIN_LIMIT): {min_pos}")
        print(f"   최대 제한값 (MAX_LIMIT): {max_pos}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    monitor_joint_limit()