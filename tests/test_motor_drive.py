import pytest

from _bootstrap import run_this_test_file  # adds project root for direct execution
from jdcobot200.hardware import MiniFeetechDriver


class FakeSerial:
    def __init__(self):
        self.writes = []
        self.is_open = True

    def write(self, packet):
        self.writes.append(bytes(packet))

    def flush(self):
        pass

    def reset_input_buffer(self):
        pass

    def read(self, count):
        return b""

    def close(self):
        self.is_open = False


def test_packet_checksum_and_sync_write_order():
    serial = FakeSerial()
    driver = MiniFeetechDriver(serial_instance=serial)

    driver.set_positions_sync({2: 2049, 1: 2048})

    packet = serial.writes[-1]
    assert packet[:5] == bytes([0xFF, 0xFF, 0xFE, 10, 0x83])
    assert packet[5:-1] == bytes([42, 2, 1, 0, 8, 2, 1, 8])
    assert packet[-1] == (~sum(packet[2:-1])) & 0xFF


def test_legacy_eeprom_offset_behavior_is_preserved():
    serial = FakeSerial()
    driver = MiniFeetechDriver(serial_instance=serial)
    driver.offsets[1] = 38

    driver.set_offset_position(1, 2048)

    packet = serial.writes[-1]
    assert packet[5:8] == bytes([42, 218, 7])  # register, 2010 little-endian


def test_position_range_is_rejected_before_serial_write():
    serial = FakeSerial()
    driver = MiniFeetechDriver(serial_instance=serial)

    with pytest.raises(ValueError, match="out of range"):
        driver.set_position(1, 4096)
    assert not serial.writes


if __name__ == "__main__":
    raise SystemExit(run_this_test_file(__file__))
