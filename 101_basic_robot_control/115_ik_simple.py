import time
import math
import numpy as np
from ikpy.chain import Chain
from motor_control import MiniFeetechDriver


PORT = "/dev/ttyUSB0"
BAUDRATE = 1000000

# 실제 로봇 팔 5축만 사용
ARM_MOTOR_IDS = [1, 2, 3, 4, 5]

URDF_PATH = "jdcobot200.urdf"

DEG_TO_TICK = 4096.0 / 360.0
RAD_TO_DEG = 180.0 / math.pi

TOTAL_STEPS = 100
STEP_DELAY = 0.02

# 중요:
# 실제 모터 방향이 URDF 방향과 반대면 -1로 바꾸세요.
JOINT_DIRECTIONS = {
    1: 1,   # shoulder
    2: 1,   # elbow
    3: 1,   # wrist_arm
    4: 1,   # wrist
    5: 1,   # gripper_base
}

# 관절별 안전 제한, degree 기준
JOINT_LIMITS_DEG = {
    1: (-120, 120),
    2: (-120, 120),
    3: (-120, 120),
    4: (-120, 120),
    5: (-180, 180),
}


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def angle_deg_to_tick_delta(motor_id, angle_deg):
    direction = JOINT_DIRECTIONS[motor_id]
    return int(direction * angle_deg * DEG_TO_TICK)


def move_smooth(driver, start_ticks, target_ticks):
    for step in range(TOTAL_STEPS + 1):
        ratio = step / TOTAL_STEPS

        # cosine easing: 부드러운 가감속
        s = 0.5 - 0.5 * math.cos(math.pi * ratio)

        for motor_id in ARM_MOTOR_IDS:
            start = start_ticks[motor_id]
            target = target_ticks[motor_id]

            pos = int(start + (target - start) * s)
            pos = clamp(pos, 0, 4095)

            driver.set_position(motor_id, pos)

        time.sleep(STEP_DELAY)


if __name__ == "__main__":
    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        # 1. IKPy 체인 생성
        robot_chain = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=["base_link"],
            last_link_vector=[0, 0, 0.05],
            active_links_mask=[
                False,  # OriginLink
                True,   # shoulder_joint
                True,   # elbow_joint
                True,   # wrist_arm_joint
                True,   # wrist_joint
                True,   # gripper_base_joint
                False,  # ripper_link
                False,  # gripper_rigiht_link
            ]
        )

        print("===== IK Chain Links =====")
        for i, link in enumerate(robot_chain.links):
            print(i, link.name)

        # 2. 현재 모터 위치 읽기
        print("\n[1] 현재 모터 위치를 기준 자세로 읽습니다.")

        center_ticks = {}

        for motor_id in ARM_MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=7)

            if pos is None:
                print(f"ID {motor_id}: 위치 읽기 실패")
                driver.close()
                exit()

            center_ticks[motor_id] = pos
            driver.set_torque(motor_id, True)
            time.sleep(0.05)

        print("기준 tick:", center_ticks)

        # 3. 목표 좌표 설정
        # 단위: meter
        target_position = np.array([0.15, 0.15, 0.01])

        print("\n[2] 목표 좌표:", target_position)

        # 4. IK 계산
        initial_position = [0] * len(robot_chain.links)

        ik = robot_chain.inverse_kinematics(
            target_position=target_position,
            initial_position=initial_position
        )

        fk = robot_chain.forward_kinematics(ik)
        actual_position = fk[:3, 3]

        print("\n[3] IK 결과 확인")
        print("target position:", target_position)
        print("actual position:", actual_position)
        print("error:", np.linalg.norm(target_position - actual_position))

        # ik[1:6] = 실제 팔 5축 각도
        joint_angles_rad = ik[1:6]
        joint_angles_deg = joint_angles_rad * RAD_TO_DEG

        print("\nIK joint angles deg:")
        for motor_id, angle in zip(ARM_MOTOR_IDS, joint_angles_deg):
            print(f"Motor {motor_id}: {angle:.2f} deg")

        # 5. 관절각을 모터 tick 목표값으로 변환
        target_ticks = {}

        for motor_id, angle_deg in zip(ARM_MOTOR_IDS, joint_angles_deg):
            min_deg, max_deg = JOINT_LIMITS_DEG[motor_id]
            angle_deg = clamp(angle_deg, min_deg, max_deg)

            delta_tick = angle_deg_to_tick_delta(motor_id, angle_deg)
            target_tick = center_ticks[motor_id] + delta_tick
            target_tick = clamp(target_tick, 0, 4095)

            target_ticks[motor_id] = target_tick

        print("\n목표 tick:", target_ticks)

        # 6. 부드럽게 이동
        print("\n[4] 목표 좌표로 이동합니다.")
        move_smooth(driver, center_ticks, target_ticks)

        print("이동 완료")

    except KeyboardInterrupt:
        print("\n사용자 종료")

    finally:
        driver.close()