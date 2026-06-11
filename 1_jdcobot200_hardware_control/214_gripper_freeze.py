import time
import serial
from motor_control import MiniFeetechDriver  # 사용자 환경에 맞는 파일명/클래스명

def freeze_gripper():
    # 통신 및 모터 설정
    PORT = "/dev/ttyACM0"   # 시리얼 포트 경로 (환경에 맞게 변경)
    BAUDRATE = 1000000     # 통신 속도 (1Mbps)
    GRIPPER_MOTOR_ID = 6   # 그리퍼 서보모터 ID [cite: 197]

    # 드라이버 초기화
    driver = MiniFeetechDriver(port=PORT, baudrate=BAUDRATE)
    
    print(f"▶ [ID: {GRIPPER_MOTOR_ID}] 그리퍼 서보 프리즈 시도...")
    
    # 그리퍼 모터만 지정하여 토크 ON (True) 
    driver.set_torque(GRIPPER_MOTOR_ID, True)
    
    print(f"✅ [ID: {GRIPPER_MOTOR_ID}] 그리퍼 토크가 잠겼습니다. 현재 위치에서 고정됩니다.")

if __name__ == "__main__":
    freeze_gripper()