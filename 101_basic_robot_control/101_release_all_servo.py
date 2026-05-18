import serial
import time
from motor_control import MiniFeetechDriver 


def main():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0'  # 윈도우라면 'COM3' 등, 리눅스라면 '/dev/ttyUSB0'
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]   # 제작하신 5축 로봇의 모터 ID 리스트
    # ----------------

    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)

    print("-" * 50)
    print("JD-101(SO-101 Clone) Bring-up: Release Mode")
    print("주의: 토크를 풀면 로봇이 주저앉을 수 있으니 손으로 잡아주세요.")
    print("-" * 50)

    # 1. 모든 모터 릴리스 (Torque Off)
    print("Releasing all servos...")
    for m_id in MOTOR_IDS:
        driver.set_torque(m_id, False)
        time.sleep(0.01) # 통신 간격 조절
    print("All servos are released. You can move the robot by hand.")

    # 2. 현재 위치 실시간 모니터링 (Ctrl+C로 종료)
    try:
        print("\nMonitoring positions (Press Ctrl+C to quit)...")
        while True:
            pos_list = []
            for m_id in MOTOR_IDS:
                pos = driver.get_position(m_id)
                if pos is not None:
                    pos_list.append(f"ID{m_id:02d}: {pos:4d}")
                else:
                    pos_list.append(f"ID{m_id:02d}: Error")
            
            # 한 줄에 출력 (\r을 사용하여 줄바꿈 없이 갱신)
            print(" | ".join(pos_list), end="\r")
            time.sleep(0.1)  # 10Hz 업데이트
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    finally:
        driver.ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()