# 테스트의 역할과 실행 방법

이 디렉터리는 실제 로봇을 움직이는 실습 프로그램이 아니라, 리팩터링한
공통 코드의 규약이 깨지지 않았는지 빠르게 확인하는 자동 테스트입니다.

## 전체 테스트 실행

저장소 최상위 디렉터리에서 다음 명령을 사용합니다.

```bash
/usr/bin/python3 -m pytest -v
```

개별 파일도 직접 실행할 수 있습니다.

```bash
/usr/bin/python3 tests/test_config.py
/usr/bin/python3 tests/test_motor_drive.py
/usr/bin/python3 tests/test_robot_description.py
```

## 각 파일의 역할

### `test_config.py`

`config/jdcobot200/robot.yaml`을 읽어 다음을 확인합니다.

- 관절 순서가 `[base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]`인지
- 모터 ID가 1~6 순서인지
- 5축 arm과 gripper 구분이 맞는지
- encoder tick과 radian 변환 방향이 맞는지
- joint limit 밖의 값을 거부하는지

### `test_motor_drive.py`

가짜 serial port를 이용하여 다음을 확인합니다.

- STS3215 명령 packet과 checksum 생성
- 여러 모터에 대한 sync-write packet
- 기존 EEPROM offset 계산 규약
- 0~4095 밖의 위치 명령 차단

가짜 serial을 사용하므로 실제 모터와 통신하거나 로봇을 움직이지 않습니다.

### `test_robot_description.py`

- canonical URDF, MJCF, 기본 scene 파일이 존재하는지
- 세 XML 파일의 기본 문법이 유효한지
- canonical mesh 디렉터리가 존재하는지

## 이 테스트가 확인하지 않는 것

현재 자동 테스트는 다음 항목을 확인하지 않습니다.

- `/dev/ttyACM0` 연결 여부
- 실제 모터 ID 응답
- 실제 encoder 값과 offset
- torque on/off
- 관절 방향과 실제 이동
- 실제 로봇과 MuJoCo의 자세 일치

이 항목은 `hardware_setup/`의 hardware smoke test 프로그램으로 확인합니다.
실제 장비 테스트는 자동 단위 테스트와 분리해야 의도하지 않은 로봇 동작을
방지할 수 있습니다.
