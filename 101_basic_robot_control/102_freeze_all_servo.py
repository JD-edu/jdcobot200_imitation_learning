import serial
import time

class MiniFeetechDriver:
    REG_GOAL_POSITION = 42
    REG_PRESENT_POSITION = 56
    REG_TORQUE_ENABLE = 40

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
            print(f"Connected to {port} at {baudrate}bps")
        except Exception as e:
            print(f"Failed to connect: {e}")
            exit()

    def _make_packet(self, motor_id, instruction, parameters):
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, motor_id, length, instruction] + parameters
        checksum = (~(sum(packet[2:]) & 0xFF)) & 0xFF
        packet.append(checksum)
        return bytearray(packet)
    
    def _write_only(self, motor_id, instruction, parameters):
        self.ser.write(self._make_packet(motor_id, instruction, parameters))

    def _write_and_read(self, motor_id, instruction, parameters, resp_bytes=8):
        self.ser.reset_input_buffer()
        self.ser.write(self._make_packet(motor_id, instruction, parameters))
        return self.ser.read(resp_bytes)

    def set_torque(self, motor_id, enable):
        self._write_only(motor_id, 0x03, [self.REG_TORQUE_ENABLE, 1 if enable else 0])

    def get_position(self, motor_id):
        resp = self._write_and_read(motor_id, 0x02, [self.REG_PRESENT_POSITION, 2], resp_bytes=8)
        if len(resp) < 8: return None
        return ((resp[6] << 8) | resp[5]) & 0x0FFF

    def set_position(self, motor_id, position):
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        self._write_only(motor_id, 0x03, [self.REG_GOAL_POSITION, pos_low, pos_high])

def main():
    # --- 설정 영역 ---
    SERIAL_PORT = '/dev/ttyUSB0' 
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
        driver.ser.close()

if __name__ == "__main__":
    main()