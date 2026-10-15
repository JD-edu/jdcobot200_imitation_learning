# JDCobot200 Imitation Learning

SO-101을 참고하여 개발한 5축 로봇암 JDCobot200 프로젝트입니다. 최종 목표는
Depth 카메라, 모방학습(Imitation Learning), 역기구학(Inverse Kinematics)을
결합한 로봇 제어 시스템과 교육과정을 만드는 것입니다.

## 폴더 구조

```text
jdcobot200_imitation_learning/
├── jdcobot200/           로봇 공통 Python 모듈
├── hardware_setup/       실제 하드웨어 점검·이동 프로그램
├── config/               로봇 설정
├── robot_description/    URDF·MJCF·mesh
├── tests/                자동 단위 테스트
├── projects/             기존 개발 과정 코드
├── docs/                 문서
├── README.md
├── pyproject.toml
└── setup.py
```

### `jdcobot200/`

다른 프로그램에서 불러 사용하는 로봇 공통 Python 모듈입니다.

```text
jdcobot200/
├── __init__.py
├── config.py             robot.yaml 로드 및 설정 검증
├── robot_description.py  canonical URDF·MJCF 경로 제공
└── hardware/
    ├── motor_drive.py    STS3215 serial packet 통신
    └── robot.py          관절 이름, 단위 변환, 제한 및 저속 이동
```

새로운 프로그램에서는 다음과 같이 사용합니다.

```python
from jdcobot200 import load_robot_config
from jdcobot200.hardware import JDCobot200Robot, MiniFeetechDriver
```

### `hardware_setup/`

실제 로봇을 연결하고 단계별로 상태와 움직임을 점검하는 실행 프로그램입니다.

```text
hardware_setup/
├── 01_freeze_servo.py       선택한 서보 고정
├── 02_release_servo.py      선택 또는 전체 서보 해제
├── 03_calibrate_servo.py    원점 offset 측정 및 저장
├── 04_move_servo_small.py   선택한 서보 10 tick 왕복
├── 05_gripper_setup.py      gripper HOME·OPEN·CLOSE 설정
└── README.md                실행 순서와 안전 지침
```

실제 로봇을 사용할 때는 반드시
[`hardware_setup/README.md`](hardware_setup/README.md)의 순서와 안전 지침을
먼저 확인합니다.

첫 단계로 서보 torque를 안전하게 해제할 수 있습니다.

```bash
/usr/bin/python3 hardware_setup/02_release_servo.py
```

### `config/`

모터 ID, 관절 순서, 회전 방향, encoder offset, joint limit, serial port 및
기본 자세를 정의합니다.

```text
config/jdcobot200/robot.yaml
```

공통 관절 및 데이터 순서는 다음과 같습니다.

```text
[base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
```

### `robot_description/`

실제 로봇과 시뮬레이션에서 공통으로 사용할 canonical 로봇 모델입니다.

```text
robot_description/
├── urdf/       URDF 로봇 모델
├── mjcf/       MuJoCo 로봇 모델과 scene
└── meshes/     STL 형상 파일
```

새로운 시뮬레이션 코드는 각 프로젝트 폴더에 모델을 복사하지 않고
`jdcobot200.robot_description`이 제공하는 경로를 사용합니다.

### `tests/`

설정, 모터 통신 packet, 단위 변환 및 로봇 모델 파일을 검사하는 자동 테스트입니다.
테스트는 가짜 serial 장치를 사용하므로 실제 로봇을 움직이지 않습니다.

```bash
/usr/bin/python3 -m pytest -v
```

각 테스트의 목적은 [`tests/README.md`](tests/README.md)에 설명되어 있습니다.

### `projects/`

하드웨어 bring-up부터 FK, IK, MuJoCo, 데이터 생성 및 모방학습까지 기존 개발
과정에서 작성한 코드가 들어 있습니다. 새로운 공통 기능을 직접 추가하는
위치가 아니라, 다음 단계의 리팩터링과 교육과정 작성을 위한 참고 공간입니다.

새로 정리한 교육용 서브프로젝트는 `projects/curriculum/`에 있습니다.

```text
projects/curriculum/
├── 01_servo_homing_and_gripper/  전체 homing과 gripper 제어
└── 02_mujoco_basics/             MuJoCo 모델 검사와 기본 제어
```

### `docs/`

코드 구성, 설계 기준 및 교육과정에 필요한 문서를 보관합니다.

### `pyproject.toml`과 `setup.py`

JDCobot200 Python 패키지의 이름, 버전 및 의존성을 정의합니다. 개발 환경에서는
다음 명령으로 설치할 수 있습니다.

```bash
python3 -m pip install -e .
```

`setup.py`는 구형 setuptools 환경과의 호환을 위해 유지합니다.

## 주의사항

- 실제 로봇을 움직이기 전에 hardware smoke test와 설정 검증을 먼저 수행합니다.
- HOME·READY 전체 이동 전에 모든 관절의 방향을 개별적으로 확인합니다.
- gripper(ID 6)는 offset을 교정하기 전까지 논리적 zero를 신뢰하지 않습니다.
- 자동 단위 테스트가 통과해도 실제 로봇의 기구적 안전을 보장하지는 않습니다.
