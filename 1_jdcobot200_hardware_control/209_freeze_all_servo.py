import serial
import time
from motor_control import MiniFeetechDriver 

def main():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyACM0' 
    MOTOR_IDS = [1, 2, 3, 4, 5]   
    # ----------------

    driver = MiniFeetechDriver(port=SERIAL_PORT)

    print("-" * 50)
    print("JD-101(SO-101 Clone) Bring-up: Freeze Mode")
    print("-" * 50)

    # 1. 안전하게 Freeze 하기
    # 현재 위치를 먼저 읽고, 그 위치를 목표값으로 설정한 뒤 토크를 켭니다.
    print("Freezing all servos at current positions...")
    
    for m_id in MOTOR_IDS:
        current_pos = driver.get_position(m_id)
        
        if current_pos is not None:
            # 현재 위치를 목표 위치로 덮어쓰기 (갑작스러운 움직임 방지)
            driver.set_position(m_id, current_pos)
            # 토크 On
            driver.set_torque(m_id, True)
            print(f"ID{m_id:02d}: Locked at {current_pos}")
        else:
            print(f"ID{m_id:02d}: Failed to read position! Check connection.")

    print("-" * 50)
    print("All servos are now FREEZED (Locked).")
    print("Press Ctrl+C to exit and keep the torque on,")
    print("or wait for the program to finish.")
    print("-" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting program. Servos will remain locked.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()