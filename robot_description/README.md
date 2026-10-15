# Canonical JDCobot200 robot description

This directory is the single source of truth for robot geometry.

- `urdf/jdcobot200.urdf`: canonical kinematic description, promoted from
  `projects/3_URDF_to_MJCF/jdcobot200.urdf` (identical to the pick-and-place copy).
- `mjcf/jdcobot200.xml`: canonical controlled MuJoCo model, promoted from
  `projects/13_dual_cam_full_cources/jdcobot200.xml` (identical to the commonly
  reused model in the historical stages).
- `mjcf/scenes/`: task-independent or task-specific worlds. A scene includes
  the robot model; cameras and task objects do not belong in the robot body.
- `meshes/`: the STL assets referenced by both formats. The small `assets` and
  `mjcf/meshes` compatibility links only resolve the different relative-path
  rules used by URDF tools and MuJoCo includes; the STL files still have one
  canonical copy.

Historical numbered directories are retained under `projects/` as
development/course snapshots.
New runtime code must use `jdcobot200.robot_description` paths rather than
copying model files.
