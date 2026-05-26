import time
from motor_control import MiniFeetechDriver

def check_all_offsets():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6] 
    
    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)

    print("="*50)
    print("JD-101 6-Axis Robot: Homing Offset Check")
    print("="*50)
    print(f"{'ID':^5} | {'Homing Offset (Stored)':^25}")
    print("-" * 50)

    for m_id in MOTOR_IDS:
        offset = driver.get_homing_offset(m_id)
        
        if offset is not None:
            print(f"ID {m_id:02d} | {offset:^25}")
        else:
            print(f"ID {m_id:02d} | {'Read Failed!':^25}")
        
        time.sleep(0.01) # 통신 안정성을 위한 짧은 대기

    print("-" * 50)
    print("확인이 완료되었습니다.")

if __name__ == "__main__":
    check_all_offsets()