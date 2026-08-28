#!/usr/bin/env bash
set -eo pipefail

DEPTH_CAM_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DEPTH_CAM_WS}/setup_env.bash"
set -u

exec ros2 launch astra_camera astra.launch.xml "$@"
