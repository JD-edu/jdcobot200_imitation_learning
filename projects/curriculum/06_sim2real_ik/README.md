# 과목 06: Sim2Real 수동 Cartesian IK

## 학습 목표

과목 04의 encoder–MuJoCo 변환과 과목 05의 Cartesian IK를 결합합니다. MuJoCo
화면에서 키보드로 TCP 이동 방향을 지정하고, FK 안전검사를 통과한 명령만 실제
로봇과 MuJoCo에 동기 적용합니다.

```text
06_sim2real_ik/
├── 01_manual_cartesian_ik.py  키보드 IK → FK 안전검사 → 실물/MuJoCo 동기 이동
├── 02_single_target_ik.py     입력 좌표의 단일 IK 해 검사 및 이동
├── 03_multi_solution_target_ik.py  여러 IK 해 탐색 후 최적 안전 경로 이동
├── target_ik_common.py        02/03 공통 FK·경로·실물 이동 검사
└── README.md
```

과목 04에서 원점·방향·회전량과 FK 오차를 먼저 검증합니다. 프로그램은 `MANUAL`
입력 후 `robot.yaml`의 READY 자세 `[0, 20, -40, 20, 0]°`로 먼저 이동하고, 실제로
측정된 READY 자세를 IK 시작점으로 사용합니다. 원점의 IK 특이 자세를 피한 상태에서
작은 이동부터 시작합니다.

```bash
/usr/bin/python3 projects/curriculum/06_sim2real_ik/01_manual_cartesian_ik.py \
  --step-mm 2 --move-time 1.5 --ready-move-time 3 \
  --feedback-tolerance-ticks 50 --max-joint-change-deg 5
```

터미널에 `MANUAL`을 입력하면 현재 자세에서 READY 자세까지의 전체 경로를 검사한 뒤
기본 3초 동안 실제 로봇과 MuJoCo를 함께 이동합니다. 이동이 끝나면 MuJoCo 창에서
조작합니다. READY 이동 시간을 바꾸려면 `--ready-move-time`을 사용합니다.

서보 목표 도달 허용오차는 기본 50 tick(4096 tick/revolution에서 약 4.39°)이며
`--feedback-tolerance-ticks`로 1~100 tick 범위에서 조정할 수 있습니다. 2초 안에 이
범위에 들지 못하면 경고와 관절 오차를 출력합니다. 이때 측정 자세가 FK 안전검사를
통과하면 실제 encoder 자세로 MuJoCo와 IK 시작점을 재동기화하여 제어를 계속합니다.
목표와 실제 관절의 차이가 `--max-joint-change-deg`보다 크면 큰 추종 실패로 보고
중단합니다. 이 허용오차는 IK의 TCP 오차 `--tolerance-mm`와 별개이며 FK 충돌·관절
제한 검사를 완화하지 않습니다.

시작 encoder 자세가 MuJoCo 관절 제한을 0.5° 이하로 살짝 벗어난 경우에는 encoder
양자화와 보정값의 경계 차이로 간주합니다. 단, 원인이 `joint_limit` 하나뿐일 때만
안전모델의 시작각을 제한 경계로 보정하여 READY 복귀 경로를 검사합니다. 0.5°보다
큰 초과나 충돌·바닥 여유 문제는 보정하지 않고 종료합니다.

공식 MJCF의 elbow actuator 하한 `-96.83°`는 `robot.yaml`에서 계산되는 실제
하드웨어 하한 약 `-113.55°`보다 좁아 Z 하강 IK가 약 150 mm 부근에서 멈춥니다.
이 과정은 elbow IK 하한을 `-110°`로 사용하여 하드웨어 끝단에 약 3.5° 여유를
남깁니다. 확장된 범위에서도 각 목표와 전체 이동 경로의 FK 충돌, 바닥 여유 및
hardware tick 검사는 그대로 수행됩니다.

- `X`, `Y`, `Z`: TCP 이동축 선택
- `LEFT`, `RIGHT`: 선택축의 음/양 방향으로 이동
- `ESC`: 종료

각 키 입력마다 현재 encoder 자세의 FK, 목표 IK 자세의 FK, 전체 관절 보간 경로,
관절 변화량과 hardware tick 범위를 검사합니다. 하나라도 통과하지 못하면 실제 모터에
명령을 보내지 않습니다. 키 1회당 이동거리는 `--step-mm`으로 0.5~10 mm 범위에서
지정할 수 있으며 기본값은 2 mm, 이동시간은 1.5초입니다.

이 검사는 소프트웨어 예방 계층입니다. 실제 운용에는 외부 장애물 확인, 저속·torque
제한, 작업자 확인과 물리 비상 정지가 별도로 필요합니다.

## 2. 입력 좌표의 단일 IK 해로 이동

현재 encoder 자세 하나만 IK 초기 seed로 사용합니다. 해가 수렴하지 않거나 목표
자세, 전체 관절 경로, 충돌·바닥 여유 또는 hardware tick 검사에서 실패하면 실제
모터 명령을 보내기 전에 종료합니다. 좌표는 MuJoCo 월드 좌표계 기준 mm입니다.

```bash
/usr/bin/python3 projects/curriculum/06_sim2real_ik/02_single_target_ik.py \
  --target-xyz-mm 0 100 300 --move-time 4
```

## 3. 여러 IK 해를 찾아 이동

현재 자세와 여러 무작위 관절 seed에서 IK를 풀고 2° 이내의 중복 해를 제거합니다.
각 해의 목표 FK와 현재 자세부터의 전체 경로를 검사한 뒤, 최대 단일 관절 변화량과
전체 관절 변화량이 가장 작은 안전한 해를 선택합니다. 기본 seed 수는 64개이며
`--seeds`로 변경할 수 있습니다.

```bash
/usr/bin/python3 projects/curriculum/06_sim2real_ik/03_multi_solution_target_ik.py \
  --target-xyz-mm 0 100 300 --seeds 64 --move-time 4
```
