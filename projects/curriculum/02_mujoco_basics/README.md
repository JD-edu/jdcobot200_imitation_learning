# 과목 02: URDF에서 안정적인 MuJoCo 모델 만들기

## 학습 목표

이 과목은 URDF를 화면에 띄우는 데서 끝나지 않습니다. URDF는 주로 로봇의 링크와
관절 형상을 설명하고, MuJoCo 시뮬레이션에는 servo의 저항·관성·제어기 설정이
추가로 필요하다는 점을 직접 비교합니다.

## 폴더 구조

```text
02_mujoco_basics/
├── 01_urdf_to_mjcf.py    URDF를 기본 MJCF로 변환
├── 02_add_dynamics.py    damping·마찰·관성·actuator 추가
├── 03_load_mjcf.py       raw/stable/canonical 모델 비교 실행
├── 04_manual_control.py  정식 MJCF 관절 수동 제어
├── scene.xml             stable 로봇에 바닥·조명을 추가한 장면
├── generated/            실행하여 생성한 MJCF
└── README.md
```

생성 파일은 원본을 덮어쓰지 않고 다음 위치에 저장됩니다.

```text
projects/curriculum/02_mujoco_basics/generated/
├── jdcobot200_raw.xml
└── jdcobot200_stable.xml
```

## 선수 과정

- Python 환경에 `mujoco` 설치
- 과목 01에서 실제 로봇의 관절 이름, HOME, 회전 방향 확인

## 1. URDF를 MJCF로 변환

```bash
/usr/bin/python3 projects/curriculum/02_mujoco_basics/01_urdf_to_mjcf.py
```

핵심 API는 다음 두 줄입니다.

```python
model = mujoco.MjModel.from_xml_path("robot.urdf")
mujoco.mj_saveLastXML("robot_raw.xml", model)
```

MuJoCo는 `<robot>` 루트인 URDF를 읽어 내부 모델로 컴파일하고,
`mj_saveLastXML`이 그 결과를 `<mujoco>` 루트인 MJCF로 저장합니다. 이 단계는 형상과
관절 구조를 옮기는 단계이지 servo 동역학을 완성하는 단계가 아닙니다.

## 2. MuJoCo 동역학 보강

```bash
/usr/bin/python3 projects/curriculum/02_mujoco_basics/02_add_dynamics.py
```

각 관절에 다음 항목을 추가합니다.

- `damping`: 속도에 비례하여 움직임을 줄이는 점성 저항
- `frictionloss`: 움직이기 시작할 때와 저속에서 작용하는 마찰
- `armature`: motor rotor와 감속기가 만드는 등가 관성
- `position actuator`: 목표 관절각을 추종하는 servo 제어기
- `forcerange`: 실제 servo가 낼 수 없는 힘을 만들지 않도록 하는 제한

또한 URDF에서 변환된 상세 STL collision mesh는 이 기초 제어 예제에서
`contype="0"`, `conaffinity="0"`으로 비활성화합니다. 겹친 상세 mesh를 그대로
충돌 계산에 사용하면 가짜 접촉력이 생겨 base가 고정되거나 시뮬레이션이 떨릴 수
있습니다. 충돌 검사는 이후 과목에서 box/capsule로 단순화한 geom을 사용합니다.

교육용 값은 STS3215를 출발점으로 한 값입니다. 실제 로봇의 step response와
비교하여 damping, frictionloss, armature, kp, kv를 다시 식별해야 합니다.

## 3. raw와 stable 모델 비교

단순 변환 모델과 동역학 보강 모델을 차례로 실행합니다.

```bash
/usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py raw
/usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py stable
```

관절 떨림, 중력에 의한 처짐, 충돌 뒤 진동, 자세 유지 능력을 비교합니다.
`stable`이 여전히 떨린다면 damping만 무작정 키우지 말고 timestep, inertia,
collision mesh, actuator `kp/kv`, force limit도 함께 확인합니다.

마지막으로 프로젝트가 사용하는 정식 MJCF scene을 엽니다.

```bash
/usr/bin/python3 projects/curriculum/02_mujoco_basics/03_load_mjcf.py canonical
```

교육용 `stable`은 변환 과정을 보여주는 최소 예제이고,
`robot_description/mjcf/scenes/empty.xml`은 이후 FK·IK·IL 과목에서 사용할 정식 모델입니다.

## 4. 정식 모델 수동 제어

```bash
/usr/bin/python3 projects/curriculum/02_mujoco_basics/04_manual_control.py
```

GLFW 창에서 `1`~`5`로 관절을 선택하고 방향키 `LEFT/RIGHT`로 5도씩 이동합니다.
`H`는 HOME, `O/C`는 gripper 열기/닫기, `ESC`는 종료입니다. 실제 로봇과
MuJoCo의 관절 양의 방향과 HOME이 같은지 확인한 뒤 다음 과목으로 넘어갑니다.

## 참고한 기존 개발 코드

- `projects/2_jdcobot200_URDF_gen`: URDF 생성·표시 과정
- `projects/3_URDF_to_MJCF`: `from_xml_path`와 `mj_saveLastXML` 변환 방식
- `projects/4_load_MJCF_to_MUJOCO`: MJCF 로드와 simulation loop

기존 코드는 개발 기록으로 유지하고, 이 과목 코드는 학습 순서에 맞게 단순화했습니다.
