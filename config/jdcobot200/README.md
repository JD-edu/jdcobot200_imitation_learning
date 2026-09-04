# JDCobot200 shared calibration

`offsets.txt` is the single canonical servo-offset file for this repository.
Each value uses the convention:

```text
offset = raw encoder tick at the calibrated home pose - 2048
```

Do not copy this file into individual example folders. Arm and gripper
calibration programs update this shared file directly.
