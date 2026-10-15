# 서브프로젝트 01: 전체 Homing과 Gripper 제어

## 학습 목표

- `robot.yaml`에서 motor ID, offset, limit을 읽는다.
- raw encoder tick과 HOME 위치의 관계를 이해한다.
- ID 1~6을 S-curve로 천천히 HOME에 이동한다.
- gripper를 OPEN, CLOSE, HOME 및 JOG 방식으로 제어한다.
- 명령 위치와 측정 위치의 오차를 확인한다.

## 폴더 구조

```text
01_servo_homing_and_gripper/
├── 01_all_servo_homing.py  ID 1~6 전체 HOME 이동
├── 02_gripper_control.py   ID 6 gripper 제어
└── README.md               학습 및 실행 안내
```

이 프로젝트에는 모터 통신 코드나 설정 파일을 복사하지 않습니다.

```text
커리큘럼 실행 코드
├── jdcobot200.hardware.MiniFeetechDriver 사용
└── config/jdcobot200/robot.yaml 사용
```

## 선수 과정

먼저 `hardware_setup`의 네 프로그램으로 다음 항목을 확인해야 합니다.

1. ID 1~6 freeze/release
2. 각 관절 HOME offset
3. 각 관절의 안전 범위
4. 각 서보의 `+10`, `-10` tick 이동

특히 `hardware_setup/05_gripper_setup.py`로 gripper의 HOME·OPEN·CLOSE를
측정하지 않았다면 전체 homing을 실행하지 마세요.

## HOME 계산

`robot.yaml`은 다음 offset 규약을 사용합니다.

```text
home_offset_tick = 2048 - raw_home_tick
raw_home_tick + home_offset_tick = 2048
```

예를 들어 offset이 `56`이면 HOME 명령은 `2104` tick입니다.

## 1. ID 1~6 전체 Homing

```bash
/usr/bin/python3 \
  projects/curriculum/01_servo_homing_and_gripper/01_all_servo_homing.py
```

프로그램은 위치 명령을 보내기 전에 모든 서보에서 다음을 출력합니다.

```text
ID, 관절 이름, 현재 tick, HOME tick
```

`HOME`을 입력해야 이동을 시작합니다. 현재 위치를 먼저 목표로 기록한 후 torque를
켜므로 이전 목표 위치로 갑자기 이동할 가능성을 줄였습니다. 이후 3초 S-curve로
ID 1~6을 함께 이동합니다.

이동 완료 후 각 서보의 목표·측정·오차를 출력합니다. 정상적으로 HOME에 도달하면
자세 유지를 위해 torque를 ON 상태로 남깁니다.

## 2. Gripper 제어

```bash
/usr/bin/python3 \
  projects/curriculum/01_servo_homing_and_gripper/02_gripper_control.py
```

메뉴에서 다음 동작을 선택할 수 있습니다.

```text
1. OPEN
2. CLOSE
3. HOME
4. +20 tick JOG
5. -20 tick JOG
6. RELEASE
0. 종료
```

OPEN과 CLOSE를 처음 확인할 때는 먼저 `+ JOG`, `- JOG`를 사용하세요. OPEN과
CLOSE 값은 `hardware_setup/05_gripper_setup.py`에서 실제 측정한 값을 사용합니다.

```python
GRIPPER_OPEN = gripper.open_tick
GRIPPER_CLOSED = gripper.closed_tick
```

OPEN과 CLOSE는 측정된 안전 범위의 min/max tick을 사용하며, JOG 명령도 해당
범위를 벗어나지 않도록 제한됩니다.

## 안전 주의사항

- 전체 homing 전에 각 관절을 개별 시험합니다.
- homing 중에는 언제든 로봇 전원을 차단할 수 있어야 합니다.
- 로봇팔을 작업대에 고정하고 이동 경로에서 사람과 물체를 치웁니다.
- gripper 구조가 변경되면 `05_gripper_setup.py`를 다시 실행합니다.
- 이상 동작이 발생하면 프로그램 종료보다 하드웨어 전원 차단을 우선합니다.
