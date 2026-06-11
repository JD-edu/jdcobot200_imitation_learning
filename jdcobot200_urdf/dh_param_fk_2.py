import numpy as np

def get_dh_matrix(a, alpha, d, theta):
    """
    표준 DH 파라미터 4가지 변수를 받아 4x4 동차 변환 행렬을 반환하는 함수
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
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
    q1, q2, q3, q4, q5 = joint_angles
    
    # [정밀 정렬된 Standard DH 테이블]
    # 각 행의 구조: [a, alpha, d, theta_offset]
    # 각 행의 회전 변수가 URDF 조인트 축의 상대적 회전과 완벽하게 매칭되도록 오프셋 가감
    dh_table = [
        [0.0,      np.pi/2,  0.0537,  0.0],       # 1. Base (수직축에서 수평축으로 전환)
        [0.1352,   0.0,      0.06146, np.pi/2],   # 2. Shoulder (홈 자세가 하늘을 향하므로 +90도 수평 정렬 오프셋 강제)
        [0.1352,   0.0,      0.0,     0.0],       # 3. Elbow 
        [0.0,      -np.pi/2, 0.0,     -np.pi/2],  # 4. Wrist Pitch (손목 정렬을 위한 롤 방향 오프셋)
        [0.0,      0.0,      0.0575,  0.0]        # 5. Wrist Roll -> EE Tip
    ]
    
    T_total = np.eye(4)
    
    for i, (a, alpha, d, theta_offset) in enumerate(dh_table):
        # 입력된 조인트 각도에 기하학적 정렬 오프셋 적용
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
    print(f"말단 XYZ 위치 (미터): X={T_end_zero[0,3]:.4f}, Y={T_end_zero[1,3]:.4f}, Z={T_end_zero[2,3]:.4f}")
    
    # 2. 어깨(q2) 90도, 팔꿈치(q3) 90도 구동 시 (대폭 변위 발생 검증)
    test_angles_move = np.radians([0.0, 90.0, 90.0, 0.0, 0.0])
    T_end_move = jdcobot200_forward_kinematics(test_angles_move)
    
    print("\n=== [테스트 2] 특정 각도 구동 시 ===")
    print(f"말단 XYZ 위치 (미터): X={T_end_move[0,3]:.4f}, Y={T_end_move[1,3]:.4f}, Z={T_end_move[2,3]:.4f}")