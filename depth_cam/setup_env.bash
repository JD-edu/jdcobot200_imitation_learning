#!/usr/bin/env bash

DEPTH_CAM_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "${DEPTH_CAM_WS}/install/setup.bash"
export LD_LIBRARY_PATH="${DEPTH_CAM_WS}/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

