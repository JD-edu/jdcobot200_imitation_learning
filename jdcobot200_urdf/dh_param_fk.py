import numpy as np

def get_dh_matrix(a, alpha, d, theta):
    """
    표준 DH 파라미터 4가지 변수를 받아 4x4 동차 변환 행렬을 반환하는 함수
    alpha와 theta는 라디안 단위여야 합니다.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    # 표준 DH 변환 행렬 공식
    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,       sa,       ca,      d],
        [ 0,        0,        0,      1]
    ])
    return T

def jdcobot200_forward_kinematics(joint_angles):
    """
    jdcobot200의 5개 관절 각도(라디안)를 받아 말단(End-Effector)의 위치와 자세를 계산
    """
    # 관절 각도 분해 (q1 ~ q5)
    q1, q2, q3, q4, q5 = joint_angles
    
    # [참고] 아래 수치는 예시이므로, 직접 유도하신 jdcobot200의 실제 DH 테이블 값으로 교체하세요.
    # 각 행의 구조: [a, alpha, d, theta_offset]
    # 실제 연산 시 theta 자리에 (qi + theta_offset)이 들어갑니다.
    dh_table = [
        [0.0,      np.pi/2,  0.0537,  0.0],  # Joint 1 (Base -> Shoulder)
        [0.1352,   0.0,      0.06146, 0.0], # Joint 2 (Shoulder -> Elbow)
        [0.1352,   0.0,      0.0,     0.0],  # Joint 3 (Elbow -> Wrist Pitch)
        [0.0,      -np.pi/2, 0.0,     0.0],  # Joint 4 (Wrist Pitch -> Wrist Roll)
        [0.0,      0.0,      0.0575,  0.0]   # Joint 5 (Wrist Roll -> End-Effector Tip)
    ]
    
    # Base 좌표계 기준 초기 행렬 (단위 행렬)
    T_total = np.eye(4)
    
    # 각 링크의 변환 행렬을 순차적으로 곱함
    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        current_theta = joint_angles[i] + theta_offset
        T_i = get_dh_matrix(a, alpha, d, current_theta)
        T_total = np.dot(T_total, T_i)
        
    return T_total

# --- 검증 및 테스트 영역 ---
if __name__ == "__main__":
    # 1. 모든 관절이 홈(Home) 자세(0도)일 때 검증
    test_angles_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    T_end_zero = jdcobot200_forward_kinematics(test_angles_zero)
    
    print("=== [테스트 1] 모든 관절 0도 (일직선 홈 포즈) ===")
    print("최종 변환 행렬 (T_end_effector):\n", np.round(T_end_zero, 4))
    print(f"말단 XYZ 위치 (미터): X={T_end_zero[0,3]:.4f}, Y={T_end_zero[1,3]:.4f}, Z={T_end_zero[2,3]:.4f}")
    
    # 2. 임의의 관절 각도를 주었을 때 연산 (예시: 라디안 단위 각도 입력)
    test_angles_move = np.radians([0.0, 1.5708, 0.0, 0.0, 0.0])
    T_end_move = jdcobot200_forward_kinematics(test_angles_move)
    
    print("\n=== [테스트 2] 특정 각도 구동 시 ===")
    print(f"말단 XYZ 위치 (미터): X={T_end_move[0,3]:.4f}, Y={T_end_move[1,3]:.4f}, Z={T_end_move[2,3]:.4f}")