import time
import numpy as np
from motor_control import MiniFeetechDriver

# ==========================================
# 1. Standard DH FK
# ==========================================
def get_dh_matrix(a, alpha, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


def jdcobot200_forward_kinematics(joint_angles_rad):
    """
    joint_angles_rad:
    사용자가 의도한 논리적 관절각.
    실제 서보 보정값은 여기 넣지 않는다.
    """

    # DH에는 기구학적으로 필요한 오프셋만 둔다
    dh_table = [
        [0.0,     np.pi/2,  0.0537,  0.0],
        [0.1352,  0.0,      0.06146, np.pi/2],
        [0.1352,  0.0,      0.0,     0.0],
        [0.0,    -np.pi/2,  0.0,    -np.pi/2],
        [0.0,     0.0,      0.0575,  0.0]
    ]

    T_total = np.eye(4)

    for i in range(5):
        a, alpha, d, theta_offset = dh_table[i]
        theta = joint_angles_rad[i] + theta_offset
        T_i = get_dh_matrix(a, alpha, d, theta)
        T_total = T_total @ T_i

    return T_total


# ==========================================
# 2. 보간 함수
# ==========================================
def smoothstep(t):
    return t * t * (3 - 2 * t)


def deg_to_rad(deg_list):
    return [np.radians(angle) for angle in deg_list]


# ==========================================
# 3. 하드웨어 보정값
# ==========================================

# 실측한 실제 서보 영점 오차
# 단위: degree
HARDWARE_ZERO_OFFSET_DEG = [
    -12.0,   # Base
    -50.0,   # Shoulder
    33.0,    # Elbow
    -81.0,   # Wrist Pitch
    0.0,     # Wrist Roll
    0.0      # Gripper
]

# 관절 방향이 반대인 경우 여기를 -1로 바꾸면 됨
MOTOR_DIRECTION = [
    1,   # Base
    1,   # Shoulder
    1,   # Elbow
    1,   # Wrist Pitch
    1,   # Wrist Roll
    1    # Gripper
]


def angle_deg_to_tick(angle_deg, motor_index, center_pos=2048):
    """
    논리적 목표각 + 하드웨어 영점 보정 → Feetech tick 변환
    """

    corrected_angle = angle_deg + HARDWARE_ZERO_OFFSET_DEG[motor_index]

    corrected_angle *= MOTOR_DIRECTION[motor_index]

    tick_offset = int(corrected_angle * (4096.0 / 360.0))
    target_tick = center_pos + tick_offset

    # 안전 범위 제한
    target_tick = max(0, min(4095, target_tick))

    return target_tick


# ==========================================
# 4. 메인 제어 루프
# ==========================================
def main():
    PORT = "/dev/ttyACM0"
    BAUDRATE = 1000000

    MOTOR_IDS = [1, 2, 3, 4, 5, 6]
    CENTER_POS = 2048

    driver = MiniFeetechDriver(PORT, BAUDRATE)
    print("▶ Feetech 스마트 서보 드라이버 연결 완료.")

    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, True)

    # 논리적 목표각
    # 여기는 보정값을 넣지 않는다
    sequence = [
        [30.0, 0.0, 0.0, 0.0,  0.0, 0.0],
        [30.0,  20.0, 20.0, 20.0, 0.0, 0.0],
        [0.0, 40.0, 40.0, 40.0,  0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    MOVE_DURATION = 2.5
    CONTROL_HZ = 50
    dt = 1.0 / CONTROL_HZ
    total_steps = int(MOVE_DURATION * CONTROL_HZ)

    print("\n🤖 jdcobot200 하드웨어 영점 보정 적용 시퀀스를 시작합니다.")
    time.sleep(1.0)

    try:
        for loop in range(2):
            print(f"\n==================== [ {loop + 1} 회 반복 시작 ] ====================")

            for step_idx, target_pose in enumerate(sequence):
                step_name = f"1-{step_idx + 1}"
                print(f"\n▶ [단계 {step_name}] 목표 각도: {target_pose}")

                for step in range(total_steps + 1):
                    t = step / total_steps
                    s = smoothstep(t)

                    interp_pose = []

                    for j in range(6):
                        start_a = current_pose[j]
                        end_a = target_pose[j]
                        interp_angle = start_a + (end_a - start_a) * s
                        interp_pose.append(interp_angle)

                    # FK는 논리적 관절각 기준으로 계산
                    interp_rad = deg_to_rad(interp_pose[:5])
                    T_end = jdcobot200_forward_kinematics(interp_rad)

                    x_cm = T_end[0, 3] * 100.0
                    y_cm = T_end[1, 3] * 100.0
                    z_cm = T_end[2, 3] * 100.0

                    if step % 10 == 0 or step == total_steps:
                        print(
                            f"[{step_name}] FK 말단 위치 -> "
                            f"X: {x_cm:6.2f}cm, "
                            f"Y: {y_cm:6.2f}cm, "
                            f"Z: {z_cm:6.2f}cm"
                        )

                    # 실제 서보 명령에는 하드웨어 보정 적용
                    for i, m_id in enumerate(MOTOR_IDS):
                        target_tick = angle_deg_to_tick(
                            interp_pose[i],
                            motor_index=i,
                            center_pos=CENTER_POS
                        )

                        driver.set_position(m_id, target_tick)

                    time.sleep(dt)

                current_pose = target_pose
                time.sleep(0.5)

        print("\n✨ 모든 시퀀스 구동 완료.")

    except KeyboardInterrupt:
        print("\n🚨 사용자 중단. 토크를 해제합니다.")

    finally:
        for m_id in MOTOR_IDS:
            driver.set_torque(m_id, False)

        print("🔒 전 관절 모터 토크 Off 완료.")


if __name__ == "__main__":
    main()