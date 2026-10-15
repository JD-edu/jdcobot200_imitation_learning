# JDCobot200 하드웨어 설정

`projects/1_jdcobot200_hardware_control`에서 실제로 사용한 제어 순서를 기준으로
만든 네 개의 간단한 프로그램입니다. 각 파일은 `MiniFeetechDriver`를 직접
사용하므로 위에서 아래로 읽으면 실행 순서를 확인할 수 있습니다.

## 파일

```text
hardware_setup/
├── 01_freeze_servo.py       선택한 서보를 현재 위치에서 고정
├── 02_release_servo.py      선택한 서보 또는 전체 서보 torque 해제
├── 03_calibrate_servo.py    선택한 서보의 원점 offset 측정 및 저장
├── 04_move_servo_small.py   선택한 서보를 10 tick 왕복 이동
├── 05_gripper_setup.py      변경된 gripper의 HOME·OPEN·CLOSE 재설정
└── README.md
```

모든 프로그램의 기본 통신 설정은 파일 위쪽에 있습니다.

```python
PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000
```

실제 포트가 다르면 실행할 파일에서 `PORT`만 수정합니다.

모터 드라이버는 공통 패키지에서 가져옵니다.

```python
from jdcobot200.hardware import MiniFeetechDriver
```

## 1. 서보 프리즈

```bash
/usr/bin/python3 hardware_setup/01_freeze_servo.py
```

서보 ID를 입력하면 현재 encoder 위치를 읽고, 그 위치를 목표로 설정한 뒤
torque를 켭니다. 프로그램이 끝나도 해당 서보의 torque는 유지됩니다.

## 2. 서보 릴리스

```bash
/usr/bin/python3 hardware_setup/02_release_servo.py
```

- ID `1~6`: 선택한 서보만 해제
- ID `0`: 전체 서보 해제

로봇팔을 손으로 지지하고 `RELEASE`를 입력해야 torque가 해제됩니다.

## 3. 서보 캘리브레이션

```bash
/usr/bin/python3 hardware_setup/03_calibrate_servo.py
```

선택한 서보의 torque를 해제한 후 관절을 손으로 원점에 맞춥니다. 프로그램은
다음 식으로 offset을 계산합니다.

```text
home_offset_tick = 2048 - raw_home_tick
```

`SAVE`를 입력하면 `config/jdcobot200/robot.yaml`에 저장하며, 기존 설정은
`robot.yaml.bak`으로 백업합니다. 측정 후 서보는 현재 위치에서 다시 고정됩니다.

## 4. 서보 미소 이동

```bash
/usr/bin/python3 hardware_setup/04_move_servo_small.py
```

서보 ID와 방향 `+1` 또는 `-1`을 선택합니다. `MOVE`를 입력하면 현재 위치에서
10 tick만 이동하고 1초 후 원위치로 돌아옵니다. 시험이 끝나면 해당 서보의
torque를 해제합니다.

```text
10 tick ≈ 0.88도
```

## 권장 실행 순서

```text
2. 서보 릴리스
→ 3. 캘리브레이션
→ 1. 서보 프리즈
→ 4. 서보 미소 이동
→ 5. 그리퍼 셋업
```

한 번에 한 서보씩 확인하고, 로봇 주변을 비운 상태에서 진행하세요.

## 5. 변경된 Gripper 셋업

```bash
/usr/bin/python3 hardware_setup/05_gripper_setup.py
```

그리퍼 torque를 해제한 뒤 다음 위치를 차례대로 손으로 맞추고 측정합니다.

1. HOME
2. 기계적 완전 OPEN
3. 기계적 완전 CLOSE

기계 끝단에서 기본 30 tick 안쪽을 안전한 OPEN/CLOSE 위치로 계산합니다. `SAVE`를
입력하면 `robot.yaml`의 ID 6 항목에 다음 값을 저장합니다.

```text
home_offset_tick
min_tick / max_tick
open_tick / closed_tick
```

저장 전 설정은 `robot.yaml.bak`으로 백업합니다. 이 과정을 마친 후 커리큘럼의
gripper control 프로그램이 측정된 OPEN/CLOSE 값을 자동으로 사용합니다.
