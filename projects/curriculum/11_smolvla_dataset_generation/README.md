# 과목 11: SmolVLA용 다양한 MuJoCo IK 데이터 생성

이 폴더는 데이터셋 생성까지만 담당합니다. 여기서 `miniVLM`은 로봇 action을 직접
출력하는 Hugging Face LeRobot의 **SmolVLA**로 해석했습니다. 일반 SmolVLM은 이미지와
텍스트를 생성하는 VLM이라 연속 관절 action policy와는 목적이 다릅니다.

Hugging Face는 SmolVLA 미세조정의 시작점으로 약 50 episode를 권장하며, 각 variation을
반복해서 담도록 안내합니다. 기본값도 50 episode입니다.

## 생성되는 데이터

09의 검증된 구성요소를 그대로 사용합니다.

- MuJoCo contact physics와 `dataset_camera` RGB(160×120)
- 안전 제한을 적용한 IK 및 전 경로 FK 검사
- 양쪽 finger 접촉, 80 mm 이상 lift, 최종 place 오차 검사
- `observation.state`: 5개 arm joint + gripper
- `action`: 5개 arm joint + gripper command
- privileged debug 값: phase, TCP 위치, block pose

추가된 다양성은 다음과 같습니다.

- pick/place 영역을 기본 5×5 격자로 나눈 층화 위치 sampling
- pick/place yaw를 기본 8개 구간으로 나눈 층화 sampling
- 09에서 물리 성공이 검증된 1.0× 시간과 smoothstep 관절 보간

모든 variation에서도 IK/FK/path safety와 실제 MuJoCo 접촉 성공 검사는 생략하지 않습니다.
접촉 기반 place 보정은 시간과 보간 함수에 민감하므로 속도/보간을 임의로 바꾸지 않고,
궤적 다양성은 작업공간 위치와 pick/place yaw 조합으로 확보합니다.

## 수동 실행

프로젝트 루트에서 다음을 실행합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/01_generate_diverse_dataset.py \
  --episodes 50 --random-seed 1100 --max-total-attempts 500
```

생성 도중 각 episode 직후 `manifest.json`을 atomic write하므로 중단 후 같은 명령을
다시 실행하면 완성된 episode는 건너뜁니다. 기존 파일을 다시 만들 때만 `--overwrite`를
사용하세요.

`--episodes 50`은 50회만 시도한다는 뜻이 아니라 **성공한 episode 50개를 채운다**는
뜻입니다. IK 또는 MuJoCo 물리 파지/배치가 실패하면 실패 seed와 원인을 manifest에
기록하고 다음 seed로 자동 재시도합니다. `--max-total-attempts`는 계속 실패할 때의
무한 반복을 방지하며, 생략하면 목표 episode 수의 10배가 적용됩니다. 중단 후 다시
실행하면 manifest의 누적 시도 횟수부터 이어지므로 같은 실패 seed를 반복하지 않습니다.
재시도 때는 seed뿐 아니라 위치 cell과 yaw bin도 함께 순환하므로 특정 물리적으로 어려운
조합에 계속 갇히지 않습니다.

기본 실행은 헤드리스입니다. 각 episode 생성 직후 실제 저장된 카메라 영상을 GUI로
확인하려면 `--preview`를 추가합니다. 미리보기에서 `Q` 또는 `ESC`를 누르면 현재까지
저장한 데이터는 유지한 채 생성 작업을 종료합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/01_generate_diverse_dataset.py \
  --episodes 50 --preview
```

더 큰 데이터셋 예시:

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/01_generate_diverse_dataset.py \
  --episodes 200 --grid-size 5 --yaw-bins 8 --random-seed 1100
```

## 검증

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/02_inspect_dataset.py
```

검사기는 모든 NPZ의 필드, frame 길이, state/action shape를 확인하고 위치 cell, yaw bin,
속도, 보간 profile별 episode 수를 출력합니다. 출력은 기본적으로
`data/train/manifest.json`과 `data/train/episodes/*.npz`에 저장됩니다.

특정 episode의 저장 영상을 GUI로 확인할 수도 있습니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/02_inspect_dataset.py \
  --preview --episode 0
```

`--episode`을 생략하면 전체 episode를 순서대로 재생합니다. `--preview-fps 10`처럼
재생 속도를 별도로 지정할 수 있으며 `Q` 또는 `ESC`로 종료합니다.

## 주의

- `11` 코드는 08/09의 scene 및 검증 로직을 import하므로 해당 폴더를 함께 유지해야 합니다.
- 이미 생성한 데이터의 fps/해상도와 다른 옵션으로 이어 만들 수 없습니다.
- 50 episode는 시작점입니다. 실제 일반화 비교에는 100~200 episode도 권장합니다.
- 이후 SmolVLA 학습 전에 이 NPZ를 LeRobot Dataset v3로 변환하면서 각 frame에 task 문장
  (예: `Pick the red block and place it on the green target.`)을 넣어야 합니다.

## NPZ를 LeRobot Dataset v3로 변환

현재 생성된 NPZ는 SmolVLA가 직접 읽을 수 없으므로 변환이 필요합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/03_convert_npz_to_lerobot.py
```

기본 출력은 `data/lerobot_jdcobot200_smolvla`입니다. 출력 폴더가 이미 있으면 기존
데이터 보호를 위해 중단합니다. 변환기는 모든 frame에 영어 task 문장을 넣고 마지막에
LeRobot으로 다시 열어 episode/frame/RGB/task를 검증합니다.

## pretrained SmolVLA 미세조정

최초 실행에는 Hugging Face에서 `lerobot/smolvla_base`와 관련 tokenizer/weight를
다운로드할 네트워크 연결이 필요합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/04_train_smolvla.py \
  --dataset-root projects/curriculum/11_smolvla_dataset_generation/data/lerobot_jdcobot200_smolvla \
  --repo-id local/jdcobot200_smolvla_pnp \
  --output projects/curriculum/11_smolvla_dataset_generation/outputs/smolvla_run_001 \
  --steps 20000 --batch-size 4 --device cuda --amp bf16
```

먼저 배선 검증만 할 때는 새 출력 경로와 `--steps 10 --batch-size 1`을 사용하세요.
VRAM 부족 시 batch size를 2 또는 1로 낮춥니다. 기본 설정은 pretrained SmolVLA의
vision/VLM을 동결하고 action expert와 state projection을 학습합니다. 5,000 step마다
checkpoint를 저장하며 최종 모델, 전처리기, 후처리기, optimizer/scheduler 상태와 학습
요약을 `pretrained_model/`에 저장합니다.

현재 설치된 LeRobot checkout은 Hub config의 `type` 필드를 제거하기 전에 concrete
config로 decode하는 버그가 있습니다. `04_train_smolvla.py`는 공식 config 내용은
유지하고 이 discriminator만 먼저 제거하여 현재 `SmolVLAConfig`로 읽습니다.

## 학습한 SmolVLA MuJoCo 추론

기본값은 GLFW 기반 3D viewer를 띄우는 GUI 방식입니다. 학습 데이터와 동일한 무작위
pick/place 장면에서 RGB, 현재 관절 상태, task 문장만 policy에 주고 폐루프로 실행합니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/05_infer_smolvla_mujoco.py \
  --model projects/curriculum/11_smolvla_dataset_generation/outputs/smolvla_run_001/pretrained_model \
  --device cuda
```

policy가 실제로 보는 `dataset_camera` 영상도 함께 보려면 `--camera-preview`를 추가합니다.
GUI 창을 닫거나 카메라 창에서 `Q`/`ESC`를 누르면 안전하게 종료합니다.
GUI는 첫 action 계산 전부터 초기 장면을 표시하고, 종료 후 기본 3초 동안 마지막 자세를
유지합니다. 유지 시간은 `--final-hold-seconds 10`처럼 조절할 수 있습니다.

창 없이 여러 episode의 성공률을 측정하고 JSON으로 저장할 수도 있습니다.

```bash
/home/robo/anaconda3/envs/lerobot/bin/python \
  projects/curriculum/11_smolvla_dataset_generation/05_infer_smolvla_mujoco.py \
  --model projects/curriculum/11_smolvla_dataset_generation/outputs/smolvla_run_001/pretrained_model \
  --device cuda --headless --episodes 10 \
  --result-json projects/curriculum/11_smolvla_dataset_generation/outputs/eval_10.json
```

기본 task는 학습 데이터와 같은
`Pick the red block and place it on the green target.`입니다. 다른 task 문장으로 학습한
모델이면 `--task "..."`로 정확히 맞춰야 합니다. action은 모델 출력 그대로 순간이동하지
않고 관절 범위와 frame당 변화량 제한(`--max-arm-step-deg`, `--max-gripper-step`)을 거쳐
실행합니다. `--seed`를 생략하면 실행할 때마다 새 seed를 만들며 출력에 사용값을 표시합니다.
같은 장면을 재현할 때만 `--seed 123`처럼 지정하세요. CPU도 `--device cpu`로 사용할 수
있지만 SmolVLA 추론은 매우 느립니다.
