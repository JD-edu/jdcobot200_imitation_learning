import time
import math
from motor_control import MiniFeetechDriver


def smoothstep(t):
    # 0~1 입력 → 0~1 출력, 시작/끝 속도 완만
    return t * t * (3 - 2 * t)


if __name__ == "__main__":
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000

    MOTOR_IDS = [1, 2, 3, 4, 5]

    LOGICAL_CENTER = 2048

    MOVE_TIME = 2.5      # 전체 이동 시간, 줄이면 더 빠름
    UPDATE_DT = 0.02     # 20ms 주기

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        start_positions = {}
        offsets = {}
        target_positions = {}

        print("[1] 현재 위치와 offset 읽기")

        for motor_id in MOTOR_IDS:
            pos = driver.get_position_filtered(motor_id, samples=5)
            offset = driver.get_homing_offset(motor_id)

            if pos is None or offset is None:
                print(f"ID{motor_id}: 읽기 실패")
                driver.close()
                exit()

            target = (LOGICAL_CENTER - offset) % 4096

            start_positions[motor_id] = pos
            offsets[motor_id] = offset
            target_positions[motor_id] = target

            driver.set_torque(motor_id, True)

            print(
                f"ID{motor_id}: raw={pos:4d}, "
                f"offset={offset:6d}, "
                f"target={target:4d}"
            )

            time.sleep(0.05)

        print("\n[2] smoothstep 프로파일로 logical 2048 위치 이동")

        steps = int(MOVE_TIME / UPDATE_DT)

        for i in range(steps + 1):
            t = i / steps
            s = smoothstep(t)

            line = []

            for motor_id in MOTOR_IDS:
                start = start_positions[motor_id]
                target = target_positions[motor_id]
                offset = offsets[motor_id]

                pos = start + (target - start) * s
                pos = int(round(pos))

                driver.set_position(motor_id, pos)

                logical = (pos + offset) % 4096
                line.append(f"ID{motor_id}:R{pos:4d}/L{logical:4d}")

                time.sleep(0.002)

            print(" | ".join(line))
            time.sleep(UPDATE_DT)

        print("\n5개 서보가 logical 2048 위치로 부드럽게 이동 완료.")

    except KeyboardInterrupt:
        print("\n사용자 중단.")

    finally:
        driver.close()