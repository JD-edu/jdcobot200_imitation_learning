# JDCobot200 합성 궤적을 LeRobot 데이터셋으로 만들기

먼저 기존 생성기로 검증된 합성 궤적을 만듭니다.

```bash
python3 generate_synthetic_trajectories.py \
  --episodes 100 \
  --control-hz 50 \
  --xml ../10_create_virtual_dataset/scene.xml \
  --output-dir synthetic_dataset
```

LeRobot을 설치하고 변환합니다.

```bash
python3 -m pip install 'lerobot>=0.4'

python3 convert_to_lerobot.py \
  --input-dir synthetic_dataset \
  --output-dir lerobot_dataset \
  --horizon 50
```

각 프레임의 주요 필드는 다음과 같습니다.

- `observation.state`: 현재 시점 `t`의 6개 실제 관절값(팔 5 + 그리퍼 1)
- `observation.images.camera`: 현재 시점 `t`의 MuJoCo RGB 이미지
- `action`: 다음 시점 `t+1`의 실제 관절값, shape `(6,)`

LeRobot은 원본 파일에는 표준 단일-step action을 저장하고, 학습 로더에서 다음과 같이 미래
50개를 `(50, 6)`으로 구성합니다. 에피소드 경계 패딩과 `action_is_pad`도 LeRobot 로더가
자동 생성합니다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

fps = 50
dataset = LeRobotDataset(
    repo_id="local/jdcobot200_pick_place",
    root="lerobot_dataset",
    video_backend="pyav",
    delta_timestamps={
        # action[t] 자체가 state[t+1]이므로 0..49가 state[t+1..t+50]입니다.
        "action": [step / fps for step in range(50)],
    },
)

sample = dataset[0]
print(sample["observation.state"].shape)          # (6,)
print(sample["observation.images.camera"].shape) # (3, 480, 640)
print(sample["action"].shape)                    # (50, 6)
print(sample["action_is_pad"].shape)             # (50,)
```

이미지는 기본 640x480, 데이터 주기는 원본 manifest의 `control_hz`입니다. 카메라 구도는
`--azimuth`, `--elevation`, `--distance`, `--lookat X Y Z`로 조절할 수 있습니다.
기존 NPZ에는 모든 MuJoCo `qpos`가 저장되므로 변환 시 그 상태를 재생하여 이미지와 관절값을
정확히 같은 프레임에서 기록합니다.

변환 스크립트는 저장이 끝난 후 위와 같은 `delta_timestamps`로 데이터셋을 다시 열고,
`action.shape == (50, 6)`인지 확인합니다. 검증 실패 시 정상 종료하지 않습니다.

현재 폴더에는 `scene.xml`이 include하는 `jdcobot200.xml`이 없기 때문에 위 생성 명령은
동일 모델이 들어 있는 이전 단계 폴더를 지정합니다. `jdcobot200.xml`을 이 폴더에 복사하면
두 스크립트 모두 `--xml`을 생략해도 현재 폴더의 `scene.xml`을 우선 사용합니다.
