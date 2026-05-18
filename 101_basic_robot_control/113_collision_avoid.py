import math
import tkinter as tk
from tkinter import messagebox

# --- 로봇 기하학 정보 설정 (jdcobot200의 실제 링크 길이에 맞게 수정 필요) ---
LINK_BASE_TO_SHOULDER = 0.10  # 베이스에서 2번 관절까지의 높이 (m)
LINK_ARM_1 = 0.15             # 2번 관절에서 3번 관절까지의 길이 (m)
LINK_ARM_2 = 0.15             # 3번 관절에서 4번 관절까지의 길이 (m)
LINK_WRIST_TO_TOOL = 0.08     # 4번 관절 이후 그리퍼 끝단까지의 총 길이 (m)

def calculate_gripper_z(angles_deg):
    """
    [순기능학(FK) 간이 계산 함수]
    입력받은 각 관절의 각도(도)를 바탕으로 그리퍼 끝단의 Z축 높이를 사전 계산합니다.
    (이해를 돕기 위해 피칭/구부러지는 주 관절인 2, 3, 4번 관절 중심의 단순화 모델 적용)
    """
    # 각도를 라디안으로 변환
    rad1 = math.radians(angles_deg[0]) # 1번 관절 (수평 회전)
    rad2 = math.radians(angles_deg[1]) # 2번 관절 (어깨 구부림)
    rad3 = math.radians(angles_deg[2]) # 3번 관절 (팔꿈치 구부림)
    rad4 = math.radians(angles_deg[3]) # 4번 관절 (손목 구부림)
    
    # 지면으로부터의 Z축 높이 누적 계산 (삼각함수 적용)
    # 로봇의 설계 도면(조인트 축 방향)에 따라 + / - 및 sin / cos 관계는 매칭이 필요합니다.
    z = LINK_BASE_TO_SHOULDER
    z += LINK_ARM_1 * math.sin(rad2)
    z += LINK_ARM_2 * math.sin(rad2 + rad3)
    z += LINK_WRIST_TO_TOOL * math.sin(rad2 + rad3 + rad4)
    
    return z

def check_and_move_robot(target_angles):
    """
    사용자가 '이동' 버튼을 눌렀을 때 실행되는 안전 검증 함수
    """
    # 1. 이동하기 전에 그리퍼가 도달할 예상 Z 높이를 먼저 계산합니다.
    predicted_z = calculate_gripper_z(target_angles)
    
    # 안전 마진 설정 (예: 바닥에서 최소 2cm(0.02m)의 여유를 둠)
    SAFETY_MARGIN_Z = 0.02 
    
    print(f"[안전 검사] 예상되는 그리퍼 끝단 높이 Z = {predicted_z:.4f} m")
    
    # 2. 기준치 이하로 내려가면 충돌 위험으로 간주하고 명령을 거부합니다.
    if predicted_z < SAFETY_MARGIN_Z:
        print("🛑 [위험] 바닥 충돌 각도 감지! 모션을 차단합니다.")
        messagebox.showerror(
            "바닥 충돌 경고", 
            f"설정하신 각도는 로봇이 바닥에 충돌할 위험이 있습니다!\n"
            f"예상 높이: {predicted_z*100:.1f} cm (안전 한계: {SAFETY_MARGIN_Z*100:.1f} cm)"
        )
        return False # 이동 불허
        
    print("✅ [안전] 충돌 위험 없음. 로봇 구동을 시작합니다.")
    # 여기에 실제 드라이버로 명령을 보내는 코드가 이어집니다 (driver.set_position 등)
    return True

# --- UI 연동 테스트부 ---
if __name__ == "__main__":
    # 예시 가상 관절 각도 [1번, 2번, 3번, 4번, 5번, 6번]
    # 2번, 3번 관절을 바닥 쪽으로 과도하게 숙인 상황 가정
    danger_pose = [0.0, -10.0, -10.0, -20.0, 0.0, 0.0]
    
    # GUI 에러 창을 띄우기 위한 임시 Tkinter 루트 생성
    root = tk.Tk()
    root.withdraw() 
    
    # 안전 검사 실행
    check_and_move_robot(danger_pose)