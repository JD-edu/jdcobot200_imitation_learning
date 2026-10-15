#!/usr/bin/env bash

PYTHON_BIN="/home/robo/anaconda3/envs/lerobot/bin/python"
SCRIPT_PATH="/home/robo/jdcobot200_imitation_learning/projects/curriculum/09_act_dataset_generation/02_generate_dataset.py"
OUTPUT_DIR="/home/robo/jdcobot200_imitation_learning/projects/curriculum/09_act_dataset_generation/data/train"
TARGET_EPISODES=100

echo "=== 에피소드 100개 수집 자동 루프 시작 ==="

while true; do
    # 현재 디렉터리에 생성된 episode_*.npz 파일 개수 세기
    CURRENT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "episode_*.npz" 2>/dev/null | wc -l)
    
    echo ""
    echo "----------------------------------------------------"
    echo " 현재 완료된 에피소드: ${CURRENT_COUNT} / ${TARGET_EPISODES}"
    echo "----------------------------------------------------"

    # 100개 이상이면 루프 종료
    if [ "$CURRENT_COUNT" -ge "$TARGET_EPISODES" ]; then
        echo " 목표 에피소드 ${TARGET_EPISODES}개가 모두 성공적으로 생성되었습니다!"
        exit 0
    fi

    # 1 ~ 100 사이에서 무작위 시드 하나 선택
    SEED=$(( (RANDOM % 100) + 1 ))
    echo ">>> 새로운 시드로 실행: random-seed = ${SEED}"

    # 에피소드 생성 파이썬 스크립트 실행
    "$PYTHON_BIN" "$SCRIPT_PATH" \
        --output "$OUTPUT_DIR" \
        --episodes "$TARGET_EPISODES" \
        --random-seed "$SEED"

    # 오류가 나더라도 스크립트가 죽지 않고 while문이 다음 시드로 계속 진행합니다.
    sleep 1
done