# 과목 07: Sim2Real Pick-and-Place

## 1. 손으로 Cartesian 좌표 티칭

실제 로봇의 arm과 gripper torque를 모두 해제한 뒤 손으로 다음 여섯 단계를
순서대로 지정합니다.

1. `pick_ready`: 열린 그리퍼로 물체를 잡을 위치에서 정지
2. `grasp`: TCP를 유지하며 그리퍼를 닫아 물체 파지
3. `lift`: 물체를 안전 높이까지 상승
4. `place_approach`: 목표 지점 상공으로 이동해 정지
5. `place`: 물체를 목표 지점 바닥까지 하강
6. `release`: TCP를 유지하며 그리퍼를 열어 물체와 완전히 분리

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/07_sim2real_pick_and_place/01_hand_teach_cartesian_points.py
```

각 자세에서 `SHOW`를 입력하면 저장하지 않고 TCP 좌표를 확인합니다. `CAPTURE`를
입력하면 FK 및 hardware 범위 검사를 통과한 자세만 기록합니다. 네 지점이 모두
완료되면 기본적으로 `taught_pick_place.json`에 TCP 위치, 회전행렬, 관절각, arm과
gripper raw encoder, 충돌·바닥 여유를 저장합니다. 저장 전 다음 조건도 검사합니다.

- `pick_ready`와 `grasp` 사이 TCP 이동거리 20 mm 이하
- `place`와 `release` 사이 TCP 이동거리 20 mm 이하
- `lift`가 `grasp`보다 Z축으로 최소 10 mm 높음
- `place_approach`가 `place`보다 Z축으로 최소 10 mm 높음

팔 torque를 해제하면 중력으로 로봇이 떨어질 수 있으므로 실행 전부터 로봇을 손으로
지지해야 합니다. 마지막에 `HOLD`를 입력한 경우에만 현재 encoder 위치를 목표로 먼저
설정한 뒤 arm과 gripper torque를 다시 켭니다. 그 외 입력이나 `ABORT` 시에는 6개
모터의 torque가 OFF인 상태로 남습니다.

## 2. 티칭한 Pick-and-Place 실행

01에서 저장한 여섯 단계 중 arm 이동에 필요한 `pick_ready`, `lift`,
`place_approach`, `place`의 Cartesian 좌표를 순서대로 IK 계획합니다. 실제 모터에
명령을 보내기 전에 전체 경로의 IK 수렴, 목표 FK, 자기충돌, 바닥 여유, 관절 변화량과
hardware tick 범위를 모두 검사합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/07_sim2real_pick_and_place/02_execute_taught_pick_and_place.py \
  --arm-move-time 5 --gripper-move-time 2 --settle-time 1
```

사전검사를 통과한 경로와 gripper tick을 확인한 후 `RUN`을 입력하면 다음 순서로
실행합니다.

1. 현재 자세에서 READY `[0, 20, -40, 20, 0]°`로 이동
2. 티칭한 `pick_ready` 값으로 gripper 열기
3. `pick_ready` TCP로 이동
4. 티칭한 `grasp` 값으로 gripper 닫기
5. `lift` TCP로 상승
6. `place_approach` TCP로 이동
7. `place` TCP로 하강
8. 티칭한 `release` 값으로 gripper 열기
9. 기본적으로 `place_approach`로 다시 상승

마지막 상승을 생략하려면 `--no-retreat`를 사용합니다. 각 arm 이동 직전에는 실제
encoder 자세에서 목표까지의 경로를 다시 검사합니다. 한 구간의 최대 관절 변화량은
기본 120°이며 `--max-joint-change-deg`로 더 작게 제한할 수 있습니다.

원점은 Cartesian position IK의 특이 자세이므로 실행 시 항상 READY 자세를 먼저
거칩니다. 모터 명령 전에는 현재 자세→READY 경로까지 포함해 전체 경로를 검사합니다.
