# IK 합성 데이터에서 LeRobot 데이터셋까지 한 번에 만들기

필요 패키지를 설치한 뒤 통합 스크립트 하나만 실행합니다.

```bash
python3 -m pip install 'lerobot>=0.4'

python3 create_lerobot_dataset.py \
  --episodes 100 \
  --control-hz 50 \
  --horizon 50 \
  --output-dir lerobot_dataset
```

이 명령은 다음 과정을 순서대로 수행합니다.

1. MINK differential IK로 pick-and-place 합성 궤적 생성
2. 매 프레임의 지면 침투, 비인접 링크 자기충돌, 관절 제한, NaN/Inf 및 배치 성공 검사
3. 실패한 후보 폐기 후 성공 에피소드 수가 찰 때까지 재생성
4. RGB 영상, 현재 관절 상태, 다음 관절 상태 action을 LeRobot 형식으로 저장
5. 저장한 데이터셋을 다시 열어 action horizon shape 검증

기본 모델은 현재 폴더에 `jdcobot200.xml`이 없으면 이전 단계의
`../10_create_virtual_dataset/scene.xml`을 자동 사용합니다. 더 보수적인 안전 여유가 필요하면
`--min-floor-clearance 0.002 --self-collision-margin 0.002`처럼 미터 단위로 지정합니다.

학습 코드에서는 다음처럼 바로 읽을 수 있습니다.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

fps = 50
dataset = LeRobotDataset(
    repo_id="local/jdcobot200_pick_place",
    root="lerobot_dataset",
    video_backend="pyav",
    delta_timestamps={"action": [i / fps for i in range(50)]},
)

sample = dataset[0]
print(sample["observation.state"].shape)           # (6,)
print(sample["observation.images.camera"].shape)  # (3, 480, 640)
print(sample["action"].shape)                      # (50, 6)
```

`synthetic_dataset/manifest.json`에는 각 에피소드의 최소 지면 여유, 자기충돌 검사 결과,
배치 오차 및 폐기 사유 집계가 남습니다. `--overwrite`는 기존 중간 데이터와 최종 데이터셋을
명시적으로 교체합니다.

## ACT 이미테이션 러닝 학습

현재 로컬 데이터셋으로 LeRobot ACT 정책을 학습합니다. CUDA가 있으면 `--device auto`가
자동 선택하며, CUDA 학습에서는 `--amp`를 함께 사용할 수 있습니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 train_lerobot_act.py \
  --dataset-root lerobot_dataset \
  --steps 20000 \
  --batch-size 8 \
  --chunk-size 50 \
  --n-action-steps 50 \
  --device auto \
  --output-dir outputs/act_jdcobot200
```

최종 모델은 다음 표준 LeRobot 디렉터리에 저장됩니다.

```text
outputs/act_jdcobot200/checkpoints/last/pretrained_model/
```

이 디렉터리에는 `config.json`, `model.safetensors`, `policy_preprocessor.json`,
`policy_postprocessor.json`과 정규화 상태가 함께 저장되므로 LeRobot의
`ACTPolicy.from_pretrained()` 및 `PolicyProcessorPipeline.from_pretrained()`로 읽을 수 있습니다.

학습된 정책을 MuJoCo에서 실행합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 infer_lerobot_act_mujoco.py \
  --checkpoint outputs/act_jdcobot200/checkpoints/last/pretrained_model \
  --dataset-root lerobot_dataset \
  --steps 500 \
  --device auto
```

기본값은 대화형 MuJoCo 뷰어를 열고 데이터셋 FPS인 50 Hz에 맞춰 실시간으로
시뮬레이션합니다. 마우스로 뷰어 카메라를 움직여도 ACT 입력에는 학습 데이터 생성 때와 같은
고정 320×240 카메라가 별도로 사용됩니다. 디스플레이가 없는 서버에서는 다음처럼 실행합니다.

```bash
MUJOCO_GL=egl /home/robo/anaconda3/envs/lerobot/bin/python3 \
  infer_lerobot_act_mujoco.py \
  --checkpoint outputs/act_jdcobot200/checkpoints/last/pretrained_model \
  --headless
```
