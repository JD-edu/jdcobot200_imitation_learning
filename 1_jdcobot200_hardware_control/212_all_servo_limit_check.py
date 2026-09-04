import os
import time
from pathlib import Path
from motor_control import MiniFeetechDriver  # 환경에 맞게 파일명을 수정하세요

# ==========================================
# [설정 영역] 포트 및 마진(버퍼) 설정
# ==========================================
PORT = "/dev/ttyACM0"      # 시리얼 포트 경로
BAUDRATE = 1000000         # 통신 속도 (1Mbps)
MOTOR_IDS = [1, 2, 3, 4, 5]  # 순차적으로 측정할 전체 모터 ID 리스트
BUFFER_VALUE = 50          # 안전을 위한 가동범위 마진 (버퍼) 값
OFFSET_FILE = Path(__file__).resolve().parents[1] / "config" / "jdcobot200" / "offsets.txt"
LIMIT_FILE = "joint_limits.txt"
# ==========================================

def load_offsets(file_path):
    """텍스트 파일에서 오프셋 값을 읽어와 딕셔너리로 반환합니다."""
    offsets = {m_id: 0 for m_id in MOTOR_IDS}  # 기본값 0으로 초기화
    if not os.path.exists(file_path):
        print(f"경고: {file_path} 파일이 없어 모든 오프셋을 0으로 시작합니다.")
        return offsets
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                m_id, offset_val = line.split("=")
                offsets[int(m_id)] = int(offset_val)
    print(f"오프셋 파일 로드 완료: {offsets}")
    return offsets

def monitor_all_joint_limits():
    # 1. 드라이버 초기화 및 오프셋 로드
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    offsets = load_offsets(OFFSET_FILE)
    
    # 최종 결과를 저장할 딕셔너리
    final_limits = {}
    
    print("\n==================================================")
    print("전체 서보모터 소프트웨어 가동범위 측정 프로그램")
    print("==================================================")
    print("각 서보의 토크를 순차적으로 해제하여 측정합니다.")
    print("측정 중인 서보 외의 다른 서보들은 자세를 유지합니다.")
    print("--------------------------------------------------")
    
    # 먼저 모든 모터의 토크를 켜서 로봇암 자세 고정
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, True)
        
    for m_id in MOTOR_IDS:
        print(f"\n[STEP] ID {m_id}번 모터 가동범위 측정 시작")
        input(f"  ID {m_id}번의 토크를 풀 준비가 되었다면 [Enter]를 누르세요...")
        
        # 해당 모터만 토크 해제
        driver.set_torque(m_id, False)
        print(f"  ID {m_id}번 모터 토크 해제 완료! 손으로 끝과 끝까지 움직여주세요.")
        print("  (측정이 완료되면 Ctrl+C를 눌러 다음 모터로 진행하세요.)")
        print("--------------------------------------------------")
        
        min_pos = 4095
        max_pos = 0
        motor_offset = offsets.get(m_id, 0)
        
        try:
            while True:
                # REG_PRESENT_POSITION(56) 레지스터에서 현재 위치 읽기
                current_pos = driver.read_u16(m_id, driver.REG_PRESENT_POSITION)
                
                if current_pos is None or current_pos < 0 or current_pos > 4095:
                    time.sleep(0.01)
                    continue
                
                # 핵심 요구사항: 오프셋이 적용된 상태의 가동범위 계산
                # 실제 제어 코드 관점에서의 가동 범위를 파악하기 위해 오프셋을 반영합니다.
                calibrated_pos = current_pos - motor_offset
                
                # 최소/최대값 실시간 갱신
                if calibrated_pos < min_pos:
                    min_pos = calibrated_pos
                if calibrated_pos > max_pos:
                    max_pos = calibrated_pos
                    
                print(f"  [실시간] raw: {current_pos:4d} | 오프셋적용: {calibrated_pos:4d} | 갱신범위: {min_pos:4d} ~ {max_pos:4d}", end="\r")
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            # 사용자가 Ctrl+C를 누르면 현재 모터 측정을 종료하고 상위 기록 유지
            print("\n--------------------------------------------------")
            print(f"  ID {m_id}번 측정 종료 (기록 저장 중...)")
            
            # 핵심 요구사항: 안전을 위한 약간의 버퍼(마진) 적용
            # 최소값은 버퍼만큼 더하고, 최대값은 버퍼만큼 빼서 더 좁고 안전한 범위를 잡습니다.
            safe_min = min_pos + BUFFER_VALUE
            safe_max = max_pos - BUFFER_VALUE
            
            # 하드웨어 물리 한계치(0~4095)를 벗어나지 않도록 클램핑
            safe_min = max(0, safe_min)
            safe_max = min(4095, safe_max)
            
            print(f"  * 순수 측정 범위: {min_pos} ~ {max_pos}")
            print(f"  * 버퍼({BUFFER_VALUE}) 적용 안전 범위: {safe_min} ~ {safe_max}")
            
            # 결과 저장
            final_limits[m_id] = (safe_min, safe_max)
            
            # 측정이 끝난 모터는 다시 토크를 켜서 고정 (로봇암 주저앉음 방지)
            print(f"  ID {m_id}번 모터 토크를 다시 켜서 자세를 고정합니다.")
            driver.set_torque(m_id, True)
            time.sleep(0.5)
            
    # 3. 모든 서보의 측정이 완료되면 텍스트 파일로 일괄 저장
    print("\n==================================================")
    print("모든 모터 측정 완료! joint_limits.txt 파일에 저장합니다.")
    print("==================================================")
    
    with open(LIMIT_FILE, "w") as f:
        for m_id in MOTOR_IDS:
            s_min, s_max = final_limits[m_id]
            f.write(f"{m_id}={s_min},{s_max}\n")
            print(f"ID {m_id} -> MIN: {s_min}, MAX: {s_max} 저장 완료")
            
    print("\n모든 작업이 성공적으로 끝났습니다. 프로그램을 종료합니다.")

if __name__ == "__main__":
    monitor_all_joint_limits()
