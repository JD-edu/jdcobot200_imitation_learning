# 과목 09: ACT 학습용 합성 데이터 생성

08에서 검증한 MuJoCo IK expert를 이용해 ACT 학습 전 단계의 원본 episode를 만듭니다.
이 과정은 **데이터 기록, 다중 episode 생성, 무결성 검사와 재생**까지만 수행합니다.
ACT 모델 변환과 학습은 다음 10 과정에서 진행합니다.

## 전체 흐름

```text
08 IK expert
  → 무작위 pick/place 위치와 yaw 생성
  → 전체 IK/FK 경로 사전검사
  → actuator 제어와 양손가락 contact/마찰로 물체 파지
  → MuJoCo RGB + 실제 관절 상태 + expert 제어 명령 동기 기록
  → NPZ episode와 JSON manifest 저장
  → 배열/시간 동기화 검사 및 영상 재생
```

MuJoCo 블록 위치와 yaw는 expert action을 만드는 데 사용되는 정답 상태입니다.
`tcp_position`과 `block_pose`에도 디버깅 목적으로 저장하지만, 10에서 ACT의 observation
입력으로 사용하면 안 됩니다. ACT가 사용할 기본 입력은 카메라 RGB와 현재 관절
상태입니다.

## 01: 단일 episode 기록

한 번의 무작위 pick-and-place를 생성하여 전체 저장 과정을 확인합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/01_record_single_episode.py
```

기본 출력 위치는 `09_act_dataset_generation/data/demo`입니다. 같은 번호의 파일을
교체할 때만 `--overwrite`를 사용합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/01_record_single_episode.py \
  --output projects/curriculum/09_act_dataset_generation/data/demo \
  --episode-index 0 --random-seed 42 --fps 20 \
  --width 160 --height 120 --overwrite
```

로봇 하드웨어에는 명령을 보내지 않으며, MuJoCo offscreen 카메라만 사용합니다.
DISPLAY가 없는 환경에서는 EGL 렌더링을 자동으로 선택합니다.

물체 pose를 TCP에 강제로 붙이지 않습니다. 양쪽 finger pad가 블록에 접촉하고, 블록을
80 mm 이상 들어 올린 뒤 목표점 35 mm 이내에 놓은 episode만 저장합니다. 이 허용오차는
`--max-place-error-mm`으로 조정할 수 있습니다.

## 02: 여러 episode 생성

서로 다른 블록 위치와 yaw를 갖는 episode를 연속 생성합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/02_generate_dataset.py \
  --episodes 100 --random-seed 42
```

각 좌표는 실행 전에 08과 동일한 IK 수렴, FK 안전성 및 전체 경로 검사를 통과해야
합니다. episode가 성공적으로 저장될 때마다 manifest도 갱신되므로 중간에 종료되어도
완료된 파일은 남습니다. 이미 존재하는 episode는 기본적으로 건너뛰며, 다시 생성하려면
`--overwrite`를 지정합니다.

동일한 데이터셋 안에서는 FPS와 영상 해상도가 같아야 합니다. 기본값은 20 FPS,
160×120 RGB입니다. 본격적인 대량 생성 전에 10~100개로 용량과 처리 시간을 확인하는
것이 좋습니다.

## 03: 데이터 검사와 재생

manifest에 등록된 모든 episode의 필드, frame 수, 배열 shape 및 timestamp 간격을
검사합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/03_inspect_dataset.py \
  --dataset projects/curriculum/09_act_dataset_generation/data/train
```

특정 episode를 화면에서 재생합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/03_inspect_dataset.py \
  --dataset projects/curriculum/09_act_dataset_generation/data/train \
  --episode 0 --replay
```

GUI 없이 MP4로 내보낼 수도 있습니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/03_inspect_dataset.py \
  --dataset projects/curriculum/09_act_dataset_generation/data/train \
  --episode 0 --export-mp4 episode_000000.mp4
```

## 저장 형식

```text
data/train/
├── manifest.json
└── episodes/
    ├── episode_000000.npz
    ├── episode_000001.npz
    └── ...
```

각 NPZ 파일에는 다음 배열이 들어 있습니다.

| 필드 | shape | 의미 |
|---|---:|---|
| `images` | `[T,H,W,3]` | `dataset_camera`의 RGB, `uint8` |
| `observation_state` | `[T,6]` | arm 5축과 gripper의 현재 radian 자세 |
| `action` | `[T,6]` | 같은 시점의 expert 관절·gripper 목표 |
| `timestamp` | `[T]` | episode 시작 기준 초 |
| `phase` | `[T]` | 동작 단계 번호 |
| `tcp_position` | `[T,3]` | 디버깅용 TCP 위치(m) |
| `block_pose` | `[T,4]` | 디버깅용 블록 `x,y,z,yaw` |

phase 번호는 다음과 같습니다.

```text
0 ready, 1 pick_approach, 2 pick, 3 grasp, 4 lift,
5 place_approach, 6 place, 7 release, 8 retreat
```

`manifest.json`에는 `generation_mode=mujoco_contact_physics`, joint 순서, FPS, 영상
크기, seed, 시작·목표 pose, 양손 접촉 여부, 최고 블록 높이와 최종 오차가 기록됩니다.
NPZ는 손실 없는 원본 보관 형식입니다. 10 과정에서 실제로 사용할 LeRobot
버전에 맞춰 이 원본을 LeRobot Dataset 형식으로 변환하고 ACT의 action chunk와 영상
전처리를 결정합니다.

## 파일 설명

- `dataset_tools.py`: 08 expert 호출, MuJoCo 렌더링, 동기 기록 및 manifest 공통 함수
- `01_record_single_episode.py`: 한 episode 생성
- `02_generate_dataset.py`: 여러 episode 생성
- `03_inspect_dataset.py`: 데이터 무결성 검사, 재생 및 MP4 내보내기
