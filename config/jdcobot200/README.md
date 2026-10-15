# JDCobot200 shared configuration

`robot.yaml` is the canonical machine-readable robot configuration. It fixes
the state/action order, motor IDs, joint directions, calibration offsets, raw
encoder limits, and serial defaults shared by hardware, simulation, IK, and IL.

The canonical order is:

```text
[base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
```

Load and validate it with `jdcobot200.load_robot_config()`. ID 6 currently has
no value in the historical `offsets.txt`, so its YAML offset is explicitly zero
until gripper calibration is performed.

## Encoder offset convention

`robot.yaml` uses the correction-to-add convention:

```text
home_offset_tick = 2048 - raw encoder tick at home
raw encoder tick at home + home_offset_tick = 2048
```

## Legacy calibration file

The former `offsets.txt` is retained at
`projects/config/jdcobot200/offsets.txt` for historical scripts. Each value uses
the convention:

```text
offset = raw encoder tick at the calibrated home pose - 2048
```

Existing calibration programs under `projects/` still update that file for
compatibility. Its values therefore have the opposite sign from
`robot.yaml.home_offset_tick`. New code should update `robot.yaml`; migration of the
interactive calibration tools is part of the next hardware-application
refactor.
