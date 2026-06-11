import serial
import time
from motor_control import MiniFeetechDriver 


def main():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyACM0'  # 윈도우라면 'COM3' 등, 리눅스라면 '/dev/ttyUSB0'
    BAUDRATE = 1000000
    MOTOR_ID = None   # 제작하신 5축 로봇의 모터 ID 리스트
    # ----------------

    driver = MiniFeetechDriver(port=SERIAL_PORT, baudrate=BAUDRATE)
    print("서보모터의 토크를 om 합니다.")
    MOTOR_ID = int(input("서보모터 번호를 입력하세요: "))

    # 1. 모든 모터 릴리스 (Torque Off)
    print("Releasing servo...")
    
    if MOTOR_ID is not None:
        driver.set_torque(MOTOR_ID, True)
    else:
        print("서보 번호가 잘못 되었습니다.")
       
    print("Servo is freezed.")

    pos = driver.get_position(MOTOR_ID)
    if pos is not None:
        print(pos, end="\r")
    else:
        print("error", end="\r")
        driver.close()

if __name__ == "__main__":
    main()