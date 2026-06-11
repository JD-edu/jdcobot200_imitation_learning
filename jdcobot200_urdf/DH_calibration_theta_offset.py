import sys
import time
import tty
import termios
import numpy as np
from motor_control import MiniFeetechDriver

# ==========================================
# 1. Standard DH 매트릭스 및 FK 수식 정의
# ==========================================
def get_dh_matrix(a, alpha, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,     sa,       ca,      d],
        [ 0,      0,        0,      1]
    ])

def jdcobot200_fk_with_offsets(joint_angles_deg, custom_offsets_deg):
    """
    사용자가 실시간으로 튜닝 중인 custom_offsets_deg를 반영하여 
    최종 End-Effector의 XYZ 위치(미터)를 계산합니다.
    """
    # [수학적 Standard DH 기본 뼈대 구조]
    # 각 행: [a, alpha, d, 수식상 필수 기본 offset]
    dh_base_table = [
        [0.0,     np.pi/2,  0.0537,  0.0],       # 1. Base
        [0.1352,  0.0,      0.06146, np.pi/2],   # 2. Shoulder
        [0.1352,  0.0,      0.0,     0.0],       # 3. Elbow
        [0.0,    -np.pi/2,  0.0,    -np.pi/2],   # 4. Wrist Pitch
        [0.0,     0.0,      0.0575,  0.0]        # 5. Wrist Roll -> EE Tip
    ]
    
    T_total = np.eye(4)
    for i in range(5):
        a, alpha, d, base_offset_rad = dh_base_table[i]
        
        # 실제 입력 각도(Degree) + 사용자가 키보드로 튜닝한 실시간 오프셋(Degree)
        total_angle_deg = joint_angles_deg[i] + custom_offsets_deg[i]
        total_angle_rad = np.radians(total_angle_deg) + base_offset_rad
        
        T_i = get_dh_matrix(a, alpha, d, total_angle_rad)
        T_total = np.dot(T_total, T_i)
        
    return T_total

# ==========================================
# 2. 리눅스 터미널 실시간 키 입력 수집 함수
# ==========================================
def getch():
    """ 엔터키 입력 없이 즉시 키보드 한 글자를 읽어오는 함수 """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# ==========================================
# 3. 정밀 튜닝 메인 루프
# ==========================================
def main():
    PORT = "/dev/ttyACM0"  # 환경에 따라 /dev/ttyUSB0 등으로 변경 [cite: 228, 324]
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]
    CENTER_POS = 2048  # Feetech 모터 원점 틱 [cite: 257]
    
    driver = MiniFeetechDriver(PORT, BAUDRATE)
    print("▶ Feetech 스마트 서보 드라이버 연결 성공.")
    
    # 안전 토크 인가 [cite: 242]
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, True) 

    # 현재 로봇에게 지시할 목표 제어 각도 (고정된 홈 포즈 테스트)
    # 튜닝할 때 로봇이 특정 각도로 가 있게 한 상태에서 오프셋을 조절하는 것이 좋습니다.
    test_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    
    # [실시간 키보드로 조절할 각 조인트별 theta_offset (도 단위)]
    # 현재 문제가 발생하는 2번 숄더에 가상의 -60도를 초기값으로 두고 시작할 수도 있습니다.
    tuned_offsets = [0.0, -60.0, 0.0, 0.0, 0.0, 0.0]
    
    # 조절 가능한 조인트 인덱스 매핑 안내
    key_map = {
        'q': (0, 1.0),  'a': (0, -1.0),   # 1축(Base)     +1도 / -1도
        'w': (1, 1.0),  's': (1, -1.0),   # 2축(Shoulder) +1도 / -1도
        'e': (2, 1.0),  't': (2, -1.0),   # 3축(Elbow)    +1도 / -1도
        'r': (3, 1.0),  'f': (3, -1.0),   # 4축(Pitch)    +1도 / -1도
        't': (4, 1.0),  'g': (4, -1.0),   # 5축(Roll)     +1도 / -1도
    }

    print("\n" + "="*60)
    print("🤖 jdcobot200 실시간 DH Theta_Offset 정밀 튜닝 모드")
    print("="*60)
    print(" [조작 키 안내] (누를 때마다 해당 조인트 오프셋 ±1도 변동)")
    print("  1축(Base)    : Q (증가) / A (감소)")
    print("  2축(Shoulder): W (증가) / S (감소)  <-- 현재 집중 튜닝 대상")
    print("  3축(Elbow)   : E (증가) / D (감소)")
    print("  4축(Pitch)   : R (증가) / F (감소)")
    print("  5축(Roll)    : T (증가) / G (감소)")
    print("  종료하려면   : ESC 또는 Ctrl+C")
    print("="*60)
    print("\n* 이제 키를 하나씩 누르면서 실제 로봇암의 기하학적 수평/수직 정렬을 맞추세요.")
    time.sleep(1.0)

    try:
        while True:
            # 1. 현재 세팅된 오프셋이 적용된 하드웨어 모터 구동 틱 계산
            for i, m_id in enumerate(MOTOR_IDS):
                sim_angle = test_joint_angles[i]
                
                # 실제 로봇에 주입할 보정 각도 = (요청 각도 + 실시간 튜닝 중인 오프셋)
                real_robot_angle = sim_angle + tuned_offsets[i]
                
                # Degree -> Feetech 틱 단위 변환
                tick_offset = int(real_robot_angle * (4096.0 / 360.0))
                target_tick = CENTER_POS + tick_offset
                driver.set_position(m_id, target_tick)
            
            # 2. 보정된 오프셋 기반 실시간 FK XYZ 계산 및 출력
            T_end = jdcobot200_fk_with_offsets(test_joint_angles[:5], tuned_offsets[:5])
            x_cm, y_cm, z_cm = T_end[0,3]*100.0, T_end[1,3]*100.0, T_end[2,3]*100.0
            
            # 터미널 한 줄 지우고 실시간 정보 플로팅
            sys.stdout.write(
                f"\r[현재 오프셋] 1축:{tuned_offsets[0]:+5.1f}° | 2축:{tuned_offsets[1]:+5.1f}° | 3축:{tuned_offsets[2]:+5.1f}° || "
                f"수학적 말단 위치 -> X:{x_cm:6.2f}cm, Y:{y_cm:6.2f}cm, Z:{z_cm:6.2f}cm"
            )
            sys.stdout.flush()
            
            # 3. 실시간 키보드 입력 대기
            ch = getch()
            
            # ESC (아스키코드 27) 누르면 종료
            if ord(ch) == 27:
                print("\n\n종료 요청을 감지했습니다.")
                break
                
            # 입력된 키에 따라 오프셋 각도 가감
            if ch in key_map:
                joint_idx, delta = key_map[ch]
                tuned_offsets[joint_idx] += delta

        print("\n" + "="*60)
        print("✨ 정밀 튜닝이 완료되었습니다! 최종 검증된 오프셋 결과:")
        print("="*60)
        for i in range(5):
            print(f" Joint {i+1} 최종 추천 theta_offset: {tuned_offsets[i]:+5.1f} 도 (라디안: {np.radians(tuned_offsets[i]):.4f} rad)")
        print("="*60)
        print("위 최종 오프셋 값을 복사하여 실전 구동 스크립트의 하드웨어 오프셋 상수로 고정하세요.")

    except KeyboardInterrupt:
        print("\n\n🚨 작업이 중단되었습니다.")
    finally:
        # 안전하게 종료 처리
        driver.close()

if __name__ == "__main__":
    main()