# 과목 05: MuJoCo IK와 안전 경로

## 학습 목표

목표 TCP XYZ를 MuJoCo Jacobian 기반 position IK로 관절각으로 바꾼 뒤, FK 오차와
시작부터 목표까지의 모든 중간 자세를 검사합니다. 이 과목은 시뮬레이션만 사용하며
하드웨어에는 연결하지 않습니다.

```text
05_mujoco_ik/
├── 01_simulation_ik.py          IK → FK → 전체 경로 안전검사
├── 02_ik_mujoco_simulation.py   검사 통과 후 MuJoCo actuator 이동 재생
├── 03_manual_cartesian_ik.py    키보드 XYZ 수동 IK와 한계 탐색
├── ik_solver.py                 damped-least-squares IK 공통 코드
└── README.md
```

먼저 시뮬레이션에서 5 mm 상대 이동을 확인합니다.

```bash
/usr/bin/python3 projects/curriculum/05_mujoco_ik/01_simulation_ik.py \
  --delta-xyz-mm 5 0 0 --start-q-deg 0 20 -40 20 0
```

검사한 IK 경로를 MuJoCo position actuator로 실제 재생합니다.

```bash
/usr/bin/python3 projects/curriculum/05_mujoco_ik/02_ik_mujoco_simulation.py \
  --target-xyz 0.05 0.10 0.35 --start-q-deg 0 20 -40 20 0

/usr/bin/python3 projects/curriculum/05_mujoco_ik/02_ik_mujoco_simulation.py \
  --delta-xyz-mm 20 20 -5 --start-q-deg 0 0 0 0 0 --move-time 4
```

최대 도달 반경, IK 수렴 오차, joint limit, 자기충돌, 바닥 여유와 전체 보간 경로를
먼저 검사합니다. 모든 검사를 통과해야 MuJoCo 창이 열리고 움직임을 시작합니다.

## 키보드로 IK 한계 탐색

```bash
/usr/bin/python3 projects/curriculum/05_mujoco_ik/03_manual_cartesian_ik.py
/usr/bin/python3 projects/curriculum/05_mujoco_ik/03_manual_cartesian_ik.py \
  --start-q-deg 0 20 -40 20 0 --step-mm 15 --move-time 0.8
```

MuJoCo 창에서 `X`, `Y`, `Z`로 이동축을 선택하고 `LEFT`, `RIGHT`로 해당 축의
음·양 방향으로 TCP를 기본 10 mm씩 이동합니다. `--step-mm`로 5~20 mm 사이의
이동 간격을 지정할 수 있습니다. `ESC`는 종료입니다. 각 입력은 최대
도달 반경, IK 수렴, FK 오차, joint limit, 자기충돌, 바닥 여유와 전체 경로 검사를
통과해야 적용됩니다. 실패한 방향과 이유는 터미널에 `거부`로 출력되므로 현재 수치
IK가 찾지 못하는 영역을 직접 탐색할 수 있습니다.

화면은 마우스 왼쪽 드래그로 회전하고 오른쪽 드래그로 이동합니다. 휠 스크롤 또는
가운데 버튼 드래그로 확대·축소할 수 있으며, `Shift`와 드래그를 함께 사용하면
수평 회전·이동으로 전환됩니다.

이 한계 탐색기는 actuator 동역학을 평가하는 코드가 아닙니다. 중력이나 torque 부족으로
명령 자세가 처지는 현상이 IK 실패로 섞이지 않도록, 안전검사를 통과한 관절 경로를
`qpos`와 `mj_forward()`로 정확하게 재생합니다. 따라서 한 번 거부된 입력은 자세를
변경하지 않으며, 즉시 다른 축이나 반대 방향을 계속 시험할 수 있습니다. actuator의
물리 추종 성능은 `02_ik_mujoco_simulation.py`에서 별도로 확인합니다.

기본 시작 관절각은 완전히 펴진 원점이 아니라 관절 운동 여유가 있는 READY 자세
`[0, 20, -40, 20, 0]°`입니다. 따라서 옵션 없이 실행해도 이 꾸부정한 자세에서 IK
탐색을 시작합니다. 다른 자세가 필요할 때만 `--start-q-deg`를 지정합니다.

IK가 수렴해도 경로 중간에 자기충돌이나 바닥 침범이 있으면 종료 코드 `2`로
거부합니다. 안전한 결과를 실제 로봇에 적용하는 과정은 과목 06에서 다룹니다.

`--tolerance-mm`는 목표를 확장하는 값이 아니라 IK 반복을 끝내는 허용오차입니다.
예를 들어 `--tolerance-mm 10`은 오차가 10 mm보다 작아지는 순간 수렴으로 판단하므로
정밀한 작업에는 사용하지 않습니다. 기본값은 1 mm이며 필요하면 0.5 mm로 낮춥니다.
또한 목표가 shoulder 기준 최대 도달 반경 밖이면 tolerance가 크더라도 계산 전에
거부합니다. 팔이 수직으로 완전히 펴진 자세에서 `+Z` 이동을 요구하는 경우가 대표적인
도달 불가능 목표입니다.
