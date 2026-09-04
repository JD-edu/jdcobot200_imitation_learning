#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/robo/jdcobot200_imitation_learning/12_data_recording_using_synth_data"
PYTHON_BIN="/home/robo/anaconda3/envs/lerobot/bin/python3"

cd "$PROJECT_DIR"
rm -f rebuild_complete.ok rebuild_failed.err validation_report.log

if "$PYTHON_BIN" convert_to_lerobot.py \
    --input-dir synthetic_dataset \
    --output-dir lerobot_dataset \
    --horizon 50 \
    --width 320 \
    --height 240 \
    --video-backend pyav \
    --image-writer-threads 8 \
    --overwrite; then
    if "$PYTHON_BIN" validate_lerobot_dataset.py \
        --root lerobot_dataset \
        --expected-episodes 50 \
        --expected-fps 50 \
        --expected-width 320 \
        --expected-height 240 \
        --horizon 50 \
        --samples 10 2>&1 | tee validation_report.log; then
        date --iso-8601=seconds > rebuild_complete.ok
        exit 0
    fi
fi

status=$?
printf 'exit_code=%s\n' "$status" > rebuild_failed.err
exit "$status"
