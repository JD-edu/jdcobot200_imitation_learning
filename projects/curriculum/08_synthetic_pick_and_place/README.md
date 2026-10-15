# 과목 08: 합성 Pick-and-Place 데이터

## 물리 기반 파지 적용 요약

이 예제는 블록을 gripper TCP에 강제로 부착하던 방식에서 MuJoCo 접촉 동역학을
사용하는 방식으로 변경되었습니다.

- 블록 위치와 자세(`qpos`)는 episode 시작 시에만 초기화합니다. 파지·운반·배치
  중에는 블록 상태를 직접 수정하지 않습니다.
- 팔과 그리퍼에는 관절 위치를 직접 대입하지 않고 position actuator의 `ctrl` 목표를
  전달합니다.
- 시뮬레이션은 `0.002 s` timestep의 `mj_step`으로 진행되어 중력, 관성, 접촉력,
  충돌과 마찰이 모두 적용됩니다.
- CAD 손가락 전체 mesh의 부정확한 convex-hull collision 대신 실제 접촉 위치에
  capsule 형태의 고무 fingertip pad를 추가했습니다.
- fingertip과 블록 사이에는 sliding friction `4.0`, torsional friction `0.30`,
  rolling friction `0.10`과 6차원 접촉(`condim=6`)을 적용했습니다.
- 그리퍼는 compliant position servo로 닫히며, 블록은 양쪽 pad의 normal force와
  마찰력만으로 들어 올립니다. weld, equality attachment 또는 좌표 동기화는
  사용하지 않습니다.
- 파지 성공에는 양쪽 fingertip 접촉과 lift 후 블록 높이 `80 mm` 이상이 필수입니다.
  release 후에는 `0.5 s` 동안 블록이 중력으로 안정화되도록 기다린 뒤 오차를
  측정합니다.

따라서 접촉 위치가 맞지 않거나 마찰력이 부족하면 블록이 실제로 밀리거나 떨어지며,
그 경우 episode는 실패합니다. 마찰 파지는 강체 결합이 아니므로 블록의 최종 yaw는
측정하여 출력하되 성공 조건으로 강제하지 않습니다.

## 1. 고정 좌표 MuJoCo Pick-and-Place

고정된 빨간 블록을 초록색 목표 위치로 옮기는 단일 IK expert episode입니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/08_synthetic_pick_and_place/01_simulated_pick_and_place.py
```

빠른 자동 검증은 viewer 없이 실행합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/08_synthetic_pick_and_place/01_simulated_pick_and_place.py \
  --no-viewer
```

실행 전에 READY부터 pick approach, pick, lift, place approach, place, retreat까지 모든
IK 목표와 전체 관절 경로의 FK 안전성을 검사합니다. 고정 블록 위치는
`[0, 180, 25] mm`, 목표 위치는 `[140, 100, 25] mm`입니다. 블록의 시작 yaw는
`30°`, 목표 yaw는 `-25°`로 설정되어 있습니다.

파지는 MuJoCo의 동역학으로 계산됩니다. 실행 중 팔과 그리퍼는 position actuator의
`ctrl` 목표만 받고, 2 ms timestep마다 `mj_step`으로 중력·관성·접촉을 적분합니다.
블록의 `qpos`는 episode 초기 위치를 설정할 때만 기록하며, 파지 이후에는 절대로
TCP에 부착하거나 위치를 덮어쓰지 않습니다.

CAD STL의 convex hull은 실제 fingertip보다 커서 블록 윗모서리에 잘못 접촉하므로,
각 손가락 body에 세로형 고무 pad collision을 추가했습니다. 블록과 pad에는 sliding,
torsional, rolling friction을 설정했습니다. 파지 직후 양쪽 pad-contact가 모두 있는지,
lift 후 블록 높이가 80 mm 이상인지 검사합니다. 놓은 뒤에는 0.5초 동안 중력으로
안정화한 실제 블록 위치를 사용해 place 오차를 계산합니다.

마찰 파지는 weld처럼 회전을 강제하지 않기 때문에 최종 yaw는 측정해 출력하지만
성공 조건으로 강제하지 않습니다. 고정 예제의 위치 허용오차는 5 mm입니다.

## 2. 무작위 위치 Pick-and-Place

매 episode마다 pick 위치와 place 위치를 서로 다른 안전 영역에서 무작위로 뽑습니다.
블록의 시작 yaw와 목표 yaw도 각각 `-90~90°`에서 무작위로 생성합니다.
무작위 좌표의 전체 IK/FK 경로가 하나라도 실패하면 해당 좌표는 실행하지 않고
폐기하며, 안전한 전체 경로를 찾을 때까지 다시 샘플링합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/08_synthetic_pick_and_place/02_randomized_pick_and_place.py \
  --episodes 5 --random-seed 42
```

GUI 없이 100개 episode를 빠르게 검증할 수 있습니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/08_synthetic_pick_and_place/02_randomized_pick_and_place.py \
  --episodes 100 --random-seed 42 --no-viewer
```

무작위 yaw에서는 블록이 손가락 안에서 회전하거나 미끄러질 수 있으므로 기본 place
허용오차는 35 mm입니다. 더 엄격한 데이터를 원하면 예를 들어
`--max-place-error-mm 10`을 지정하십시오. 양손 접촉과 실제 lift 검사는 허용오차와
무관하게 항상 필수입니다.

기본 무작위 영역은 다음과 같습니다.

```text
pick:  X=-55~55 mm,  Y=155~205 mm, Z=25 mm
place: X=105~175 mm, Y=65~135 mm,  Z=25 mm
```

`--random-seed`가 같으면 동일한 위치 순서를 다시 생성하므로 실패 episode를 재현할
수 있습니다. 이 단계는 아직 이미지나 action을 파일로 저장하지 않으며, 다양한
좌표에서 expert trajectory가 안정적으로 생성되는지를 검증합니다.

## Wrist-roll과 블록 yaw 정렬

01과 02 모두 MuJoCo가 가진 블록 quaternion에서 yaw를 알고 있는 합성 expert입니다.
각 waypoint의 위치 IK를 구한 뒤, FK 관절 제한 안에서 `wrist_roll` 후보 361개를
검사하여 손가락 닫힘 축이 블록의 긴 축과 수직이 되는 해를 선택합니다. 실제 양쪽
손가락 형상이 180° 회전 대칭이 아니므로 닫힘 축의 방향까지 구분하며, yaw가 거의
같은 후보에서는 현재 자세와 가장 가까운 `wrist_roll`을 선택해 불필요한 큰 회전을
줄입니다.

pick 구간에서는 블록의 시작 yaw를, 운반 후 place 구간에서는 목표 yaw를 그리퍼
목표로 사용합니다. 실제 블록 yaw는 접촉과 마찰의 결과이며 gripper yaw에 강제로
동기화하지 않습니다. yaw는 expert trajectory를 생성하기 위한 MuJoCo의 privileged
state입니다. 이후 ACT 학습 입력에는 이 값을 직접 제공하지 않고 카메라 RGB
관측에서 블록 방향을 학습하도록 구성해야 합니다.
