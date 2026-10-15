# 과목 04: 실제 로봇과 MuJoCo 동기 제어

## 목적

실제 로봇과 MuJoCo가 같은 관절 이름, 원점, 회전 방향을 사용하도록 확인한 뒤 두
로봇을 같은 관절각 명령으로 움직입니다. 실제 이동 명령보다 항상 먼저 과목 03의
joint limit, 자기충돌, 바닥 여유와 전체 보간 경로를 검사합니다.

## Sim2Real 관절 정합 조건

실제 로봇과 MuJoCo가 같은 관절 자세를 나타내려면 다음 조건을 모두 확인해야 합니다.

1. **원점 일치**
   - 실제 encoder에서 `raw_tick + home_offset_tick = 2048`이면 0도입니다.
   - 이때 실제 관절과 MuJoCo 관절의 물리적 자세가 같아야 합니다.
2. **회전 방향 일치**
   - MuJoCo 관절각이 증가할 때 실제 관절도 같은 물리적 방향으로 움직여야 합니다.
   - 실제 모터의 CW/CCW와 MuJoCo 관절축의 양의 방향을 비교합니다.
3. **회전량 일치**
   - MuJoCo가 10도 움직이면 실제 관절도 10도 움직여야 합니다.
   - STS3215의 4096 tick/revolution 기준 변환식은 다음과 같습니다.

     ```text
     corrected_tick = raw_tick + home_offset_tick
     joint_degree = direction * (corrected_tick - 2048) * 360 / 4096
     ```

4. **관절 순서 일치**

   ```text
   1=base, 2=shoulder, 3=elbow, 4=wrist_pitch, 5=wrist_roll
   ```

5. **기구 형상과 TCP 일치**
   - MJCF의 링크 길이, 관절축 위치·방향과 TCP 위치가 실제 로봇과 같아야 합니다.
   - 엔코더 각도가 같아도 MJCF의 `q=0` 형상이 다르면 두 로봇의 자세는 일치하지
     않습니다.
6. **관절 제한 일치**
   - `robot.yaml`의 실제 encoder 안전 범위와 MJCF의 joint/actuator range가 서로
     일치해야 합니다.

즉, 같은 원점과 방향만으로는 충분하지 않습니다. **원점, 방향, 회전량, 관절 순서,
기구 형상, TCP 및 관절 제한**이 모두 일치해야 관절 위치 기준 Sim2Real 정합이
완료된 것으로 판단합니다. 이 과목의 01은 원점·방향·회전량을 확인하고, 02와 03은
동기 이동과 전체 경로 안전검사를 확인합니다.

```text
04_sim2real_fk/
├── 01_calibrate_joint_direction.py  실물/MuJoCo 방향 비교와 robot.yaml 갱신
├── 02_sync_joint_move.py            지정 관절 동기 이동
├── 03_sync_manual_control.py        키보드 동기 수동 제어
├── sim2real_control.py              변환·viewer·안전 동기 이동 공통 코드
└── README.md
```

## 1. 관절 방향 확인

한 번에 하나의 관절을 기본 10도 왕복시킵니다. 먼저 실제 encoder 5개의 현재값을
읽고 `corrected_tick = raw_tick + home_offset_tick`으로 보정합니다. 보정값 2048이
MuJoCo의 0도이며, 이 관절각으로 MuJoCo의 초기 자세를 표시합니다. 이후 화면의 MuJoCo와 실물의 회전 방향이
같은지 관찰하여 `SAME` 또는 `OPPOSITE`를 입력합니다. `UPDATE`까지 입력한 경우에만
`config/jdcobot200/robot.yaml`의 해당 `direction`을 저장하며, 기존 파일은
`robot.yaml.bak`으로 백업합니다. 1번부터 5번까지 각각 확인합니다.

```bash
/usr/bin/python3 projects/curriculum/04_sim2real_fk/01_calibrate_joint_direction.py --joint 1
/usr/bin/python3 projects/curriculum/04_sim2real_fk/01_calibrate_joint_direction.py \
  --joint wrist_pitch --test-deg 10
```

## 2. 특정 관절 동기 이동

관절은 번호 1~5 또는 이름으로 지정합니다. `--delta-deg`는 현재 자세 기준 상대각,
`--target-deg`는 MuJoCo 논리 절대각입니다. `--all`을 사용하면 관절 순서
`base shoulder elbow wrist_pitch wrist_roll`에 해당하는 각도 5개를 받아 모든 관절을
동시에 움직입니다. 전체 경로가 안전하고 작업자가 `MOVE`를 입력한 경우에만 실제
로봇과 화면의 MuJoCo가 함께 움직입니다.

관절별 1회 변화량 제한은 단일 관절에서 기본 10도, `--all`에서 기본 45도입니다.
필요하면 `--max-change-deg`로 더 작게 제한할 수 있습니다.

```bash
/usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py \
  --joint 2 --delta-deg 3
/usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py \
  --joint wrist_pitch --target-deg 10 --move-time 4
/usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py \
  --all --delta-deg 2 -3 4 -2 1
/usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py \
  --all --target-deg 0 20 -40 20 0 --max-change-deg 45 --move-time 5
```

안전검사는 세 번 적용됩니다. `MOVE`를 묻기 전에 전체 관절 보간 경로를 검사하고,
실행 직전에 같은 경로를 다시 검사하며, 모터로 보내는 각 중간 자세도 다시 검사합니다.

## 3. 키보드 동기 수동 제어

시작할 때 `MANUAL`을 입력한 뒤 GLFW 창에서 조작합니다.

- `1`~`5`: 관절 선택
- `LEFT`, `RIGHT`: 선택 관절을 기본 5도 이동
- `ESC`: 종료

```bash
/usr/bin/python3 projects/curriculum/04_sim2real_fk/03_sync_manual_control.py
/usr/bin/python3 projects/curriculum/04_sim2real_fk/03_sync_manual_control.py \
  --step-deg 5 --move-time 1
```

각 키 입력마다 새 목표까지의 전체 경로를 먼저 검사합니다. 위험한 입력은 터미널에
`REJECTED`를 출력하고 실제 모터에는 전송하지 않습니다. 소프트웨어 검사와 별도로
저속·torque 제한, 작업자 확인과 물리 비상 정지가 필요합니다.

목표 도달 확인은 2초 안에 모든 관절이 목표의 15 encoder tick 이내에 들어왔는지
검사합니다. 부하나 서보의 정상상태 오차로 이 조건을 만족하지 못하면 경고를
출력하고, 실제 encoder 자세를 다시 읽어 MuJoCo 및 다음 이동의 시작 자세로
사용합니다. 같은 관절에서 경고가 반복되거나 오차가 커지면 조작을 중단하고 기구물
간섭, 전원 전압, 서보 부하와 torque 설정을 확인해야 합니다.
