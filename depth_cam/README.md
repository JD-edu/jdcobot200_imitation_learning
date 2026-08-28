# Orbbec Astra NB bring-up (ROS 2 Humble)

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

