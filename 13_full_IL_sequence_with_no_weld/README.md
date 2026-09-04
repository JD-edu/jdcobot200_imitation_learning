# JDCobot200 MuJoCo contact-based imitation learning

이 디렉터리는 **weld/equality로 물체를 붙이지 않고**, MuJoCo 양쪽 finger pad의 실제 contact를 확인한 뒤 compliant grip force를 적용하는 전체 파이프라인입니다.

파이프라인은 다음 네 단계로 구성됩니다.

1. MuJoCo에서 contact 기반 pick-and-place 시연(NPZ) 생성
2. front/wrist RGB와 joint state/action을 LeRobot v3 데이터셋으로 변환
3. LeRobot ACT policy 학습
4. 동일한 contact 모델에서 closed-loop 추론 및 결과 JSON 저장

## 환경

검증에 사용한 인터프리터는 `/home/robo/anaconda3/envs/lerobot/bin/python3`입니다. `mujoco`, `numpy`, `torch`, `lerobot`, FFmpeg/PyAV가 필요합니다. headless 렌더링에서는 `MUJOCO_GL=egl`을 사용합니다.

## 1. 데이터 획득과 LeRobot 변환

```bash
MUJOCO_GL=egl /home/robo/anaconda3/envs/lerobot/bin/python3 \
  create_lerobot_dataset.py --episodes 50 --width 320 --height 240 \
  --frame-stride 5 --overwrite
```

원본 50 Hz trajectory는 `synthetic_dataset/`에 남고, 10 Hz dual-camera LeRobot 데이터는 `lerobot_dataset/`에 생성됩니다. 후보 episode는 finite value, joint limit, floor clearance, self-collision, 양쪽 finger 접촉, 30 mm 이상 lift, 목표 배치 오차를 통과해야 저장됩니다. 기본 `-8 mm` clearance 허용치는 바닥과 충돌하지 않도록 mask된 아래쪽 fingertip collision proxy의 형상 범위 때문입니다.

검증:

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 validate_lerobot_dataset.py \
  --root lerobot_dataset --repo-id local/jdcobot200_contact_pick_place \
  --expected-episodes 50 --expected-width 320 --expected-height 240
```

빠른 smoke test는 `--episodes 1 --width 160 --height 120 --frame-stride 5`로 실행할 수 있습니다.

## 2. ACT 학습

```bash
/home/robo/anaconda3/envs/lerobot/bin/python3 train_lerobot_act.py \
  --dataset-root lerobot_dataset \
  --repo-id local/jdcobot200_contact_pick_place \
  --steps 20000 --batch-size 8 --num-workers 0 \
  --chunk-size 50 --n-action-steps 50 --device auto \
  --output-dir outputs/act_jdcobot200
```

checkpoint는 `outputs/act_jdcobot200/checkpoints/last/pretrained_model/`에 저장됩니다.

## 3. MuJoCo closed-loop 추론

```bash
MUJOCO_GL=egl /home/robo/anaconda3/envs/lerobot/bin/python3 \
  infer_lerobot_act_mujoco.py --headless --no-realtime --steps 500
```

`outputs/inference_report.json`에 bilateral-contact frame 수, grasp/release step, 최종 block pose, 목표 오차와 성공 여부가 기록됩니다. GUI를 보려면 `--headless`를 제거합니다.

## contact 구현 원칙

- `scene.xml`에 block-gripper weld가 없으며 런타임에도 equality를 활성화하지 않습니다.
- 좌/우 pad와 block geom이 실제 MuJoCo contact pair를 형성해야 grasp가 시작됩니다.
- 접촉 이후에는 최대 3 N의 spring-damper force로 마찰식 grasp를 안정화합니다. 물체 pose를 순간 이동시키거나 고정하지 않습니다.
- gripper open 명령에서 force가 즉시 해제되고 이후 물체는 중력과 contact dynamics만 따릅니다.
- 생성된 manifest와 inference report에는 `uses_weld: false`가 명시됩니다.

전체 명령을 순서대로 실행하려면 `run_full_pipeline.sh`를 사용합니다. 인자는 episode 수와 학습 step 수입니다.

```bash
./run_full_pipeline.sh 50 20000
```
