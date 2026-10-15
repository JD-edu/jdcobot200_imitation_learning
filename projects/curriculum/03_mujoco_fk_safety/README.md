# 과목 03: MuJoCo FK와 안전 자세 검사

## 학습 목표

관절각 5개를 공식 MuJoCo 모델에 적용하고 `mj_forward()`로 TCP 위치와 방향을
계산합니다. 같은 FK 결과를 이용해 다음 위험을 이동 전에 검사합니다.

- joint limit 초과
- 서로 이웃하지 않은 링크 사이의 자기충돌 여유
- shoulder부터 gripper까지의 바닥 여유
- 시작 자세에서 목표 자세까지 중간 경로의 위험

상세 STL mesh는 충돌 계산이 느리고 겹침 오류에 민감하므로 cylinder, sphere,
capsule로 만든 보수적인 `collision_proxy`를 사용합니다. proxy 크기는 실제 로봇을
측정하면서 계속 조정해야 합니다.

## 파일 순서

```text
03_mujoco_fk_safety/
├── 01_complete_mjcf.py       stable MJCF에 TCP·충돌 proxy 추가
├── 02_fk_pose.py             한 자세의 TCP 위치·방향 계산
├── 03_check_pose.py          자세 또는 관절 경로 안전 검사
├── 04_safe_manual_control.py 안전한 명령만 적용하는 GLFW 예제
├── fk_safety.py              공통 FK·거리 검사 코드
├── scene.xml                 완성 후보 모델과 바닥을 결합
└── README.md
```

## 1. FK에 필요한 MJCF 완성

현재 FK와 안전검사는 검증을 마친 공식 모델
`robot_description/mjcf/jdcobot200.xml`을 사용합니다. 아래 생성 코드는 모델 변환
과정을 학습하거나 새 후보를 만들 때만 사용하며, 생성 후보를 Sim2Real 기준 모델로
직접 사용하지 않습니다.

과목 02의 `jdcobot200_stable.xml`에는 관절 동역학과 actuator는 있지만, URDF에
정의되지 않았던 TCP와 안전검사용 링크 부피가 없습니다. 다음 코드가 이를 추가하여
`generated/jdcobot200_complete.xml`을 만듭니다.

```bash
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/01_complete_mjcf.py
```

추가되는 핵심 항목은 다음과 같습니다.

- `graspframe`: FK·IK가 사용할 TCP 위치와 방향
- `collision_*`: 자기충돌과 바닥 거리를 계산할 단순 proxy 6개
- joint, actuator, equality, mesh 경로의 로드 여부 검증

현재 TCP는 `gripper_assembly` 원점에서 `0 0 -0.08 m`입니다. 이 값은 예시가
아니라 실제 로봇의 집기 중심을 실측하여 확정해야 합니다. TCP 길이를 바꾸면
`collision_gripper`의 `fromto`도 함께 바뀌도록 코드가 구성되어 있습니다.

## 2. FK 계산

관절 순서는 `base shoulder elbow wrist_pitch wrist_roll`이고 입력 단위는 degree입니다.

```bash
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/02_fk_pose.py 0 20 -40 20 0
```

FK 계산은 물리 시간을 진행하는 `mj_step()`이 아니라 현재 자세의 좌표만 갱신하는
`mujoco.mj_forward()`를 사용합니다.

## 3. 목표까지 전체 경로 검사

HOME에서 READY 자세까지 101개 중간 자세를 검사합니다.

```bash
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/03_check_pose.py 0 20 -40 20 0
```

다른 시작 자세를 지정할 수도 있습니다.

```bash
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/03_check_pose.py \
  0 30 -50 20 0 --start-deg 0 20 -40 20 0 --steps 201
```

안전하면 종료 코드 `0`, 위험하면 `2`를 반환합니다. 자동화 코드에서는 출력 문자열이
아니라 이 종료 코드를 사용할 수 있습니다.

## 4. GLFW 안전 수동 제어

```bash
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/04_safe_manual_control.py
```

`1`~`5`로 관절을 선택하고 `LEFT/RIGHT`로 5도씩 움직입니다. 현재 목표에서 새
목표까지 21개 중간 자세가 모두 안전할 때만 명령을 적용합니다. 위험한 명령은
이유와 자기충돌·바닥 여유를 출력하고 거부합니다. `ESC`로 종료합니다.

## 5. 검증한 후보를 공식 XML로 승격

이 과목의 FK, 자기충돌, 바닥 검사와 실제 로봇 TCP 측정이 모두 끝난 뒤
`generated/jdcobot200_complete.xml`의 내용을 공식 모델에 반영합니다.

공식 모델 위치는 다음과 같습니다.

```text
robot_description/mjcf/jdcobot200.xml
```

먼저 후보를 공식 XML 폴더에 다른 이름으로 복사하여 비교합니다. 기존 공식 파일을
곧바로 덮어쓰면 안 됩니다.

```bash
cp projects/curriculum/03_mujoco_fk_safety/generated/jdcobot200_complete.xml \
   robot_description/mjcf/jdcobot200_candidate.xml
diff -u robot_description/mjcf/jdcobot200.xml \
        robot_description/mjcf/jdcobot200_candidate.xml
```

아래 검증 항목을 모두 통과한 변경만 `jdcobot200.xml`에 반영합니다. 최종적으로
`jdcobot200.xml`이 공식 단일 모델이며, `jdcobot200_candidate.xml`은 비교가 끝나면
제거합니다.

후보 파일을 그대로 복사하기 전 다음 항목을 반드시 비교해야 합니다.

- 실제 TCP 위치와 방향
- 실제 joint limit과 회전 방향
- 링크 질량과 관성
- damping, frictionloss, armature
- actuator `kp`, `kv`, `forcerange`
- collision proxy가 실제 로봇 전체를 충분히 감싸는지

후보의 검증된 변경을 공식 XML에 반영한 후 다음 명령으로 차이를 확인합니다.

```bash
git diff -- robot_description/mjcf/jdcobot200.xml
/usr/bin/python3 projects/curriculum/03_mujoco_fk_safety/02_fk_pose.py 0 0 0 0 0
```

`generated` 후보를 자동 복사하지 않는 이유는 잘못 측정한 TCP나 임시 물리값으로
공식 모델을 덮어쓰는 것을 막기 위해서입니다. 이후 FK·IK·IL 코드는 검증이 끝난
`robot_description/mjcf/jdcobot200.xml`을 공식 기준으로 사용합니다.

또한 생성 후보의 `meshdir`는 예제 실행을 위해 현재 PC의 절대경로입니다. 공식
XML에는 이 값을 복사하지 말고 기존의 이식 가능한 `meshdir="../meshes/"`를
유지해야 합니다.

## 주의

이 장치는 소프트웨어의 1차 예방 계층입니다. 실제 로봇에는 별도로 저속 제한,
torque 제한, timeout, 비상 정지와 작업자 확인이 필요합니다.
