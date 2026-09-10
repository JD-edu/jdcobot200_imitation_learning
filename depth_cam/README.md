# Orbbec Astra NB bring-up

## Python 뷰어 (ROS 2 불필요)

작업공간에 포함된 Orbbec OpenNI2 런타임을 직접 사용하므로 ROS 2 노드나
토픽을 실행하지 않습니다. OpenCV 창에서 `q` 또는 `Esc`를 누르면 종료됩니다.

```bash
# RGB + Depth (기본)
./run_astra_python.bash

# 각각 단독 실행
./run_astra_python.bash --stream color
./run_astra_python.bash --stream depth
./run_astra_python.bash --stream ir

# Depth + IR
./run_astra_python.bash --stream depth-ir
```

해상도/FPS와 depth 표시 최대 거리도 변경할 수 있습니다.

```bash
./run_astra_python.bash --stream depth --width 640 --height 480 --fps 30 --max-depth 5000
```

> Astra 하드웨어 제약상 Color와 IR은 동시에 실행할 수 없습니다. 첫 실행에서
> 권한 오류가 나면 아래의 USB permission setup을 먼저 적용합니다.

## ROS 2 Humble

The connected camera is detected as `2bc5:0401 Orbbec Astra`. The workspace
contains the Orbbec ROS 2 Astra driver, its OpenNI2 runtime, and a locally built
libuvc.

## One-time USB permission setup

Run this from the workspace root, then unplug and reconnect the camera:

```bash
sudo cp src/ros2_astra_camera/astra_camera/scripts/56-orbbec-usb.rules /etc/udev/rules.d/56-orbbec-usb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Bring up

```bash
./bringup_astra.bash
```

In another terminal:

```bash
source setup_env.bash
ros2 topic list
ros2 topic hz /camera/depth/image_raw
```

To view the depth image:

```bash
source setup_env.bash
rqt_image_view /camera/depth/image_raw
```
