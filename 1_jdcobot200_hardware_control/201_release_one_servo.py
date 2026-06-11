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
    print("서보모터의 토크를 off 합니다.")
    MOTOR_ID = int(input("서보모터 번호를 입력하세요: "))

    # 1. 모터 릴리스 (Torque Off)
    print("Releasing servo...")
    
    if MOTOR_ID is not None:
        driver.set_torque(MOTOR_ID, False)
    else:
        print("서보 번호가 잘못 되었습니다.")
       
    print("Servo is released. You can move the robot by hand.")

    # 2. 현재 위치 실시간 모니터링 (Ctrl+C로 종료)
    try:
        print("\nMonitoring positions (Press Ctrl+C to quit)...")
        while True:
            pos = driver.get_position(MOTOR_ID)
            if pos is not None:
                print(pos, end="\r")
            else:
                print("error", end="\r")
            time.sleep(0.1)  # 10Hz 업데이트
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()