import time
import math
import numpy as np
from ikpy.chain import Chain
from motor_control import MiniFeetechDriver


PORT = "/dev/ttyACM0"
BAUDRATE = 1000000

ARM_MOTOR_IDS = [1, 2, 3, 4, 5]

URDF_PATH = "jdcobot200.urdf"

DEG_TO_TICK = 4096.0 / 360.0
RAD_TO_DEG = 180.0 / math.pi

TOTAL_STEPS = 80
STEP_DELAY = 0.02


JOINT_DIRECTIONS = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
}


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def angle_deg_to_tick_delta(motor_id, angle_deg):
    direction = JOINT_DIRECTIONS[motor_id]
    return int(direction * angle_deg * DEG_TO_TICK)


def move_smooth(driver, start_ticks, target_ticks):
    for step in range(TOTAL_STEPS + 1):

        ratio = step / TOTAL_STEPS

        # cosine easing
        s = 0.5 - 0.5 * math.cos(math.pi * ratio)

        for motor_id in ARM_MOTOR_IDS:

            start = start_ticks[motor_id]
            target = target_ticks[motor_id]

            pos = int(start + (target - start) * s)

            pos = clamp(pos, 0, 4095)

            driver.set_position(motor_id, pos)

        time.sleep(STEP_DELAY)


def ik_to_ticks(ik_angles_deg, center_ticks):

    target_ticks = {}

    for motor_id, angle_deg in zip(ARM_MOTOR_IDS, ik_angles_deg):

        delta_tick = angle_deg_to_tick_delta(
            motor_id,
            angle_deg
        )

        target_tick = center_ticks[motor_id] + delta_tick

        target_tick = clamp(target_tick, 0, 4095)

        target_ticks[motor_id] = target_tick

    return target_ticks


if __name__ == "__main__":

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:

        # ---------------------------------------------------
        # IK 체인 생성
        # ---------------------------------------------------
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
                False,  # gripper_left
                False,  # gripper_right
            ]
        )

        # ---------------------------------------------------
        # 현재 위치 읽기
        # ---------------------------------------------------
        center_ticks = {}

        for motor_id in ARM_MOTOR_IDS:

            pos = driver.get_position_filtered(
                motor_id,
                samples=7
            )

            if pos is None:
                print(f"motor {motor_id} read fail")
                exit()

            center_ticks[motor_id] = pos

            driver.set_torque(motor_id, True)

            time.sleep(0.05)

        print("\ncenter ticks =", center_ticks)

        # ---------------------------------------------------
        # 현재 IK 기준 자세
        # ---------------------------------------------------
        previous_ik = [0] * len(robot_chain.links)

        # ---------------------------------------------------
        # 기준 위치
        # ---------------------------------------------------
        base_target = np.array([
            0.15,
            0.15,
            0.00
        ])

        # 먼저 기준 위치로 이동
        ik = robot_chain.inverse_kinematics(
            target_position=base_target,
            initial_position=previous_ik
        )

        previous_ik = ik

        joint_deg = np.degrees(ik[1:6])

        target_ticks = ik_to_ticks(
            joint_deg,
            center_ticks
        )

        move_smooth(
            driver,
            center_ticks,
            target_ticks
        )

        current_ticks = target_ticks

        time.sleep(1)

        print("\n===== Z AXIS TEST START =====")

        # ---------------------------------------------------
        # z축 +5cm
        # ---------------------------------------------------
        up_target = base_target + np.array([
            0.0,
            0.0,
            0.05
        ])

        print("\nmove up")
        print("target =", up_target)

        ik_up = robot_chain.inverse_kinematics(
            target_position=up_target,
            initial_position=previous_ik
        )

        previous_ik = ik_up

        fk_up = robot_chain.forward_kinematics(ik_up)

        print("actual =", fk_up[:3, 3])

        joint_deg_up = np.degrees(ik_up[1:6])

        print("angles =", joint_deg_up)

        up_ticks = ik_to_ticks(
            joint_deg_up,
            center_ticks
        )

        move_smooth(
            driver,
            current_ticks,
            up_ticks
        )

        current_ticks = up_ticks

        time.sleep(2)

        # ---------------------------------------------------
        # z축 -5cm
        # ---------------------------------------------------
        down_target = base_target + np.array([
            0.0,
            0.0,
            -0.05
        ])

        print("\nmove down")
        print("target =", down_target)

        ik_down = robot_chain.inverse_kinematics(
            target_position=down_target,
            initial_position=previous_ik
        )

        previous_ik = ik_down

        fk_down = robot_chain.forward_kinematics(ik_down)

        print("actual =", fk_down[:3, 3])

        joint_deg_down = np.degrees(ik_down[1:6])

        print("angles =", joint_deg_down)

        down_ticks = ik_to_ticks(
            joint_deg_down,
            center_ticks
        )

        move_smooth(
            driver,
            current_ticks,
            down_ticks
        )

        current_ticks = down_ticks

        time.sleep(2)

        # ---------------------------------------------------
        # 다시 기준 위치 복귀
        # ---------------------------------------------------
        print("\nreturn center")

        ik_center = robot_chain.inverse_kinematics(
            target_position=base_target,
            initial_position=previous_ik
        )

        joint_deg_center = np.degrees(
            ik_center[1:6]
        )

        center_return_ticks = ik_to_ticks(
            joint_deg_center,
            center_ticks
        )

        move_smooth(
            driver,
            current_ticks,
            center_return_ticks
        )

        print("\nDONE")

    except KeyboardInterrupt:
        print("\nSTOP")

    finally:
        driver.close()