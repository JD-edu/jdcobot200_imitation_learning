import serial
import time

class MiniFeetechDriver:
    # --- 주요 레지스터 주소 ---
    REG_ID = 5
    REG_MIN_POSITION_LIMIT = 9
    REG_MAX_POSITION_LIMIT = 11
    REG_HOMING_OFFSET = 31
    REG_TORQUE_ENABLE = 40
    REG_GOAL_POSITION = 42
    REG_LOCK = 55            # EEPROM 쓰기 잠금 레지스터
    REG_PRESENT_POSITION = 56

    def __init__(self, port='/dev/ttyUSB0', baudrate=1000000):
        self.ser = serial.Serial(port, baudrate, timeout=0.05)
        # 각 모터의 오프셋 값을 저장할 메모리 공간
        self.offsets = {}

    def load_all_offsets(self, motor_ids):
        """설정된 모든 모터의 오프셋을 읽어와 self.offsets에 캐싱합니다."""
        for m_id in motor_ids:
            val = self.get_homing_offset(m_id)
            self.offsets[m_id] = val if val is not None else 0
        print(f"[*] 오프셋 로드 완료: {self.offsets}")

    def set_offset_position(self, motor_id, logical_position):
        """
        사용자가 설정한 원점을 기준으로 이동합니다.
        물리적 위치 = (목표 논리 위치 - 저장된 오프셋) % 4096
        """
        offset = self.offsets.get(motor_id, 0)
        
        # 1. 물리적 목표값 계산
        physical_position = logical_position - offset
        
        # 2. [핵심] 0~4095 범위를 순환하도록 처리 (Modulo 연산)
        # 이렇게 하면 -1이 4095가 되고, 4096이 0이 되어 끊김 없이 연결됩니다.
        physical_position = physical_position % 4096
        print(motor_id, physical_position)
        
        # 3. 정수형 변환 및 최종 전송
        self.set_position(motor_id, int(physical_position))

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

    # ---- EEPROM 잠금 해제 및 잠금 (핵심 추가) ----
    def unLockEprom(self, motor_id):
        """EEPROM 영역 수정을 위해 잠금을 해제합니다 (0 기록)"""
        # 0x03: WRITE instruction
        self._write_only(motor_id, 0x03, [self.REG_LOCK, 0])
        time.sleep(0.01) # 처리를 위한 짧은 대기

    def lockEprom(self, motor_id):
        """설정 완료 후 EEPROM 영역을 보호합니다 (1 기록)"""
        self._write_only(motor_id, 0x03, [self.REG_LOCK, 1])
        time.sleep(0.01)

    # ---- Helper functions ----
    def write_u16(self, motor_id, reg_addr, value):
        """2바이트(16비트) 데이터를 기록합니다 (Little Endian)"""
        lo = value & 0xFF
        hi = (value >> 8) & 0xFF
        self._write_only(motor_id, 0x03, [reg_addr & 0xFF, lo, hi])

    def read_u16(self, motor_id, reg_addr, retries=3):
        for _ in range(retries):
            resp = self._write_and_read(
                motor_id,
                0x02,
                [reg_addr & 0xFF, 2],
                resp_bytes=8
            )

            if len(resp) == 8 and self._check_packet(resp, motor_id):
                error = resp[4]
                if error != 0:
                    return None

                return (resp[6] << 8) | resp[5]

            time.sleep(0.01)

        return None
    
    def _check_packet(self, resp, motor_id):
        if len(resp) < 6:
            return False

        if resp[0] != 0xFF or resp[1] != 0xFF:
            return False

        if resp[2] != motor_id:
            return False

        checksum = (~(sum(resp[2:-1]) & 0xFF)) & 0xFF
        if checksum != resp[-1]:
            return False

        return True

    # ---- 캘리브레이션 및 설정 기능 ----
    def set_homing_offset(self, motor_id, offset):
        """오프셋을 기록합니다. (반드시 호출 전 unLockEprom 필요)"""
        # 오프셋은 음수일 수 있으므로 16비트 signed 처리를 위해 & 0xFFFF 사용
        self.write_u16(motor_id, self.REG_HOMING_OFFSET, int(offset) & 0xFFFF)

    def set_id(self, current_id, new_id):
        """모터의 ID를 변경합니다. (반드시 호출 전 unLockEprom 필요)"""
        self._write_only(current_id, 0x03, [self.REG_ID, new_id])
        time.sleep(0.2) # ID 변경 후 메모리 갱신 대기

    # ---- 제어 기능 ----
    def set_torque(self, motor_id, enable):
        self._write_only(motor_id, 0x03, [self.REG_TORQUE_ENABLE, 1 if enable else 0])

    def set_position(self, motor_id, position):
        self.write_u16(motor_id, self.REG_GOAL_POSITION, position)

    def get_position(self, motor_id):
        pos = self.read_u16(motor_id, self.REG_PRESENT_POSITION)

        if pos is None:
            return None

        pos = pos & 0x0FFF

        if pos < 0 or pos > 4095:
            return None

        return pos
    
    def get_position_filtered(self, motor_id, samples=5):
        values = []

        for _ in range(samples):
            pos = self.get_position(motor_id)
            if pos is not None:
                values.append(pos)
            time.sleep(0.01)

        if not values:
            return None

        values.sort()
        return values[len(values) // 2]
    
    def get_homing_offset(self, motor_id):
        """
        모터의 Homing Offset 레지스터(31번 주소) 값을 읽어옵니다.
        STS3215는 16비트 Signed 데이터를 사용하므로 음수 처리가 필요합니다.
        """
        # 31번 주소(REG_HOMING_OFFSET)에서 2바이트를 읽어옵니다.
        raw_offset = self.read_u16(motor_id, self.REG_HOMING_OFFSET)
        
        if raw_offset is None:
            return None
        
        # 16비트 Signed Integer 변환 로직
        # 0x8000(32768) 이상이면 음수로 변환합니다.
        if raw_offset > 0x7FFF:
            raw_offset -= 0x10000
            
        return raw_offset
    
import time


import time
from motor_control import MiniFeetechDriver


if __name__ == "__main__":
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]

    driver = MiniFeetechDriver(PORT, BAUDRATE)

    try:
        while True:
            line = []

            for motor_id in MOTOR_IDS:
                pos = driver.get_position_filtered(motor_id, samples=5)
                offset = driver.get_homing_offset(motor_id)

                if pos is None or offset is None:
                    line.append(f"ID{motor_id}: ----")
                else:
                    total = pos + offset
                    line.append(f"ID{motor_id}:{total:5d}")

                time.sleep(0.02)

            print(" | ".join(line))

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n종료합니다.")

    finally:
        driver.close()