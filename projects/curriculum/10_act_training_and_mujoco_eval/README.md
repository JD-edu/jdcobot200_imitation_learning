# 과목 10: ACT 학습과 MuJoCo 평가

09의 원본 NPZ 데이터는 변경하지 않습니다. 첫 단계에서 별도 변환기를 사용하여 현재
설치된 LeRobot `0.4.4`가 직접 읽을 수 있는 Dataset v3 형식으로 변환합니다.

## 01: 09 원본을 LeRobot Dataset v3로 변환

먼저 09에서 원본 데이터를 생성합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/09_act_dataset_generation/02_generate_dataset.py \
  --episodes 100 --random-seed 42
```

그다음 아직 존재하지 않는 새로운 출력 경로로 변환합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/10_act_training_and_mujoco_eval/01_convert_npz_to_lerobot.py \
  --source projects/curriculum/09_act_dataset_generation/data/train \
  --output projects/curriculum/10_act_training_and_mujoco_eval/data/jdcobot200_pnp \
  --repo-id local/jdcobot200_synthetic_pnp
```

변환 결과에는 LeRobot 표준의 `meta/`, `data/`, `images/` 구조와 통계 파일이
생성됩니다. 변환기는 마지막에 `LeRobotDataset`으로 결과를 다시 열어 episode 수,
frame 수와 RGB tensor shape까지 검사합니다.

ACT가 입력으로 사용하는 필드는 다음 세 가지입니다.

- `observation.images.dataset_camera`: RGB
- `observation.state`: 5개 arm 관절과 gripper
- `action`: 다음 시점에 실행할 5개 arm 관절과 gripper 목표

`debug.block_pose`, `debug.tcp_position`, `debug.phase`, `next.done`,
`next.success`도 보존하지만 이름이 `observation.*` 또는 `action`이 아니므로 LeRobot
ACT policy feature에는 포함되지 않습니다.

출력 경로가 이미 존재하면 변환기는 기존 데이터를 보호하기 위해 중단합니다.
다시 변환하려면 기존 폴더를 직접 보관하거나 새로운 출력 폴더를 사용하세요.

## 02: LeRobot ACT 학습

`02_train_act.py`는 별도의 ACT 모델을 재구현하지 않습니다. 설치된 LeRobot 0.4.4의
`LeRobotDataset`, `ACTConfig`, `ACTPolicy`, 공식 pre/post processor와 optimizer
구성을 그대로 사용합니다. 최종 모델과 정규화 processor도 LeRobot
`save_pretrained()` 형식으로 함께 저장합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/10_act_training_and_mujoco_eval/02_train_act.py \
  --dataset-root projects/curriculum/10_act_training_and_mujoco_eval/data/jdcobot200_pnp \
  --repo-id local/jdcobot200_synthetic_pnp \
  --output projects/curriculum/10_act_training_and_mujoco_eval/outputs/act_run_001 \
  --steps 100000 --batch-size 8 --chunk-size 50 --n-action-steps 25 \
  --device cuda --amp
```

현재 설치의 공식 `lerobot-train` CLI는 ACT와 관계없는 GR00T 설정을 import하는 중
Python dataclass 오류가 발생합니다. 이 코드는 그 진입점만 우회하고 ACT 학습에
사용하는 LeRobot 공식 클래스와 처리 순서는 그대로 유지합니다. 이 우회는
`lerobot.policies` package의 ACT 하위 경로만 등록하는 것이며 설치된 LeRobot 파일을
수정하지 않습니다.

### 중요한 학습 옵션

- `--steps`: 업데이트 횟수입니다. 먼저 1,000~5,000 step으로 loss와 추론 배선을
  확인한 뒤 100,000 step을 권장합니다.
- `--batch-size`: GPU 메모리에 가장 큰 영향을 줍니다. 8에서 시작하고 OOM이면 4나
  2로 줄입니다.
- `--chunk-size`: 한 관측으로 학습할 미래 action 길이입니다. 20 FPS에서 50은
  2.5초입니다. 한 episode의 frame 수보다 지나치게 크지 않아야 합니다.
- `--n-action-steps`: 추론 시 한 번 예측한 chunk에서 실제 사용할 action 수입니다.
  작으면 자주 재계획하고, 크면 빠르지만 오차에 늦게 대응합니다.
- `--learning-rate`: ACT 기본값은 `1e-5`입니다. 처음에는 변경하지 않는 것이 좋습니다.
- `--amp`: CUDA 메모리와 시간을 줄이는 혼합 정밀도입니다. CUDA에서만 사용합니다.
- `--device`: `auto`, `cuda`, `mps`, `cpu` 중 하나입니다. 현재 장비에서 CUDA가
  인식되는지 먼저 확인해야 하며, `auto`는 사용 가능한 장치를 자동 선택합니다.
- `--num-workers`: 데이터 로딩 프로세스 수입니다. 기본 4이며 메모리가 부족하면 0~2로
  줄입니다.
- `--no-pretrained-backbone`: ImageNet ResNet18 초기값을 사용하지 않습니다. 일반
  학습에서는 지정하지 않는 편이 좋고, 네트워크 없는 smoke test에만 유용합니다.
- `--dim-model` 등의 모델 크기 옵션은 정상 학습에서는 기본값을 유지합니다. 작은
  모델은 코드 검증에는 빠르지만 최종 성능 비교 기준이 달라집니다.

학습 결과는 다음 구조로 저장됩니다.

```text
outputs/act_run_001/
├── checkpoints/020000/pretrained_model/
├── checkpoints/040000/pretrained_model/
└── pretrained_model/
    ├── config.json
    ├── model.safetensors
    ├── policy_preprocessor.json
    ├── policy_postprocessor.json
    ├── training_state.pt
    └── training_summary.json
```

가장 중요한 점은 학습 데이터의 양과 다양성입니다. 코드 동작 확인에는 수십 episode도
가능하지만, 위치·yaw·조명·색상 변화에 일반화하려면 최소 수백 episode부터 평가하는
것이 적절합니다. `debug.block_pose` 같은 privileged state는 policy feature로 전달되지
않습니다.

## 03: 학습한 ACT를 MuJoCo에서 평가

학습이 끝난 `pretrained_model`과 함께 저장된 전처리/후처리를 불러와, 데이터 생성 때와
동일한 `dataset_camera`, 관절 순서, pick/place 분포에서 closed-loop 평가합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/10_act_training_and_mujoco_eval/03_eval_act_mujoco.py \
  --model projects/curriculum/10_act_training_and_mujoco_eval/outputs/act_run_002/pretrained_model \
  --device cuda
```

실행하면 MuJoCo 창이 열리고 블록과 초록색 목표가 무작위로 배치됩니다. 이후에는 expert
경로나 정답 관절값을 사용하지 않고, 매 순간 `dataset_camera` 영상과 현재 관절 상태를
ACT에 넣어 얻은 action만으로 로봇이 움직입니다. 결과가 실패하더라도 모델이 예측한
동작을 그대로 확인할 수 있습니다.

ACT에 실제로 들어가는 160×120 영상을 함께 보려면 `--camera-preview`를 추가합니다.
창 없이 여러 episode의 성공률만 계산하려면 `--headless --episodes 10`을 사용합니다.
기본 300 step은 학습 데이터의 약 273 frame보다 조금 긴 추론 시간입니다. 매 episode마다
`policy.reset()`을 호출하므로 ACT action queue가 이전 episode에서 넘어오지 않습니다.

평가기는 합성 데이터 생성기와 같은 방식으로 정렬 조건을 만족한 gripper에 블록을
부착하고, 열린 gripper에서 분리합니다. 성공 조건은 실제 파지와 분리가 모두 발생하고
최종 위치 오차 5 mm 및 yaw 오차 5도 이내인 경우입니다. 관절별 frame 변화 제한은
비정상 모델 출력에 대한 안전장치이며 `--max-arm-step-deg`와
`--max-gripper-step`으로 조정할 수 있습니다.
