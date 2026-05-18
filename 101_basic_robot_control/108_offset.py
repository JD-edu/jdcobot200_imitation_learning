import time
from motor_control import MiniFeetechDriver


def calibrate_and_save_offset():
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000

    # 그리퍼 제외 5축이면 [1,2,3,4,5]
    # 6축 전체면 [1,2,3,4,5,6]
    MOTOR_IDS = [1, 2, 3, 4, 5]

    LOGICAL_CENTER = 2048

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        print("=" * 70)
        print("STS3215 Homing Offset Calibration")
        print("=" * 70)

        print("\n[1] 토크 OFF")
        for motor_id in MOTOR_IDS:
            driver.set_torque(motor_id, False)
            time.sleep(0.03)

        print("\n로봇을 손으로 원하는 중심 자세에 맞추세요.")
        input("중심 자세를 맞춘 후 [Enter]를 누르세요: ")

        print("\n[2] 현재 엔코더값 기준으로 offset 계산")
        print("-" * 70)

        calculated_offsets = {}

        for motor_id in MOTOR_IDS:
            raw_pos = driver.get_position_filtered(motor_id, samples=7)

            if raw_pos is None:
                print(f"ID{motor_id}: 엔코더 읽기 실패")
                calculated_offsets[motor_id] = None
                continue

            # logical = raw + offset
            # 따라서 logical 2048이 되려면:
            # offset = 2048 - raw
            offset = LOGICAL_CENTER - raw_pos

            # signed 16bit 범위 보정
            if offset > 32767:
                offset -= 65536
            elif offset < -32768:
                offset += 65536

            check_logical = raw_pos + offset

            calculated_offsets[motor_id] = offset

            print(
                f"ID{motor_id}: "
                f"raw={raw_pos:4d}, "
                f"offset={offset:6d}, "
                f"raw+offset={check_logical:4d}"
            )

            time.sleep(0.05)

        print("-" * 70)
        input("위 raw+offset 값이 모두 2048인지 확인 후 EEPROM 저장하려면 [Enter]: ")

        print("\n[3] EEPROM에 Homing Offset 저장")
        print("-" * 70)

        for motor_id in MOTOR_IDS:
            offset = calculated_offsets[motor_id]

            if offset is None:
                print(f"ID{motor_id}: offset 없음, 저장 건너뜀")
                continue

            try:
                driver.unLockEprom(motor_id)
                time.sleep(0.05)

                driver.set_homing_offset(motor_id, offset)
                time.sleep(0.05)

                driver.lockEprom(motor_id)
                time.sleep(0.05)

                print(f"ID{motor_id}: offset {offset} EEPROM 저장 완료")

            except Exception as e:
                print(f"ID{motor_id}: EEPROM 저장 실패 - {e}")

        print("\n[4] EEPROM 저장값 다시 읽기")
        print("-" * 70)

        for motor_id in MOTOR_IDS:
            saved_offset = driver.get_homing_offset(motor_id)
            raw_pos = driver.get_position_filtered(motor_id, samples=7)

            if saved_offset is None or raw_pos is None:
                print(f"ID{motor_id}: readback 실패")
                continue

            logical = raw_pos + saved_offset

            print(
                f"ID{motor_id}: "
                f"raw={raw_pos:4d}, "
                f"saved_offset={saved_offset:6d}, "
                f"raw+offset={logical:4d}"
            )

        print("\n완료되었습니다.")
        print("전원을 껐다 켠 뒤 다시 raw+offset이 2048 근처인지 확인하세요.")

    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        driver.close()


if __name__ == "__main__":
    calibrate_and_save_offset()