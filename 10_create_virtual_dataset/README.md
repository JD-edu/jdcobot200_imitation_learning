# JDCobot200 synthetic demonstrations

`generate_synthetic_trajectories.py` creates reproducible, fixed-rate MuJoCo
pick-and-place demonstrations.  The red block and robot joint pose start at a
different valid position in every episode.  Random safe-height waypoints,
heights, and segment durations provide multiple paths to one fixed goal.

Generate the requested 100 trajectories:

```bash
python3 generate_synthetic_trajectories.py --episodes 100
```

Every candidate is replayed headlessly and every moving robot mesh vertex is
checked on every frame. Candidates with floor penetration, joint-limit
violations, non-finite state, or failed placement are rejected. The initial
candidate budget is 2x the requested episode count; generation continues past
that budget when necessary until the requested number of valid episodes has
been collected. To keep the previous dataset and create a newly validated one:

```bash
python3 generate_synthetic_trajectories.py --episodes 100 \
  --output-dir synthetic_dataset_safe
```

Use `--min-floor-clearance 0.002` to require an additional 2 mm margin. The
manifest records the minimum clearance and limiting robot geom for every
accepted episode, along with aggregate rejection counts.

Use `--seed` for another reproducible dataset and `--overwrite` to replace an
existing output directory.  A quick smoke test is:

```bash
python3 generate_synthetic_trajectories.py --episodes 1 \
  --output-dir /tmp/jdcobot_smoke
```

Each `episode_NNNN.npz` contains arrays at 50 Hz:

- `timestamp`: simulation time, `(T,)`
- `qpos`, `qvel`: complete MuJoCo state, `(T, nq)` and `(T, nv)`
- `action`: five arm position commands plus gripper command, `(T, 6)`
- `ee_position`, `desired_ee_position`: actual/reference tool XYZ, `(T, 3)`
- `block_position`, `block_quaternion`: object pose, `(T, 3)` and `(T, 4)`
- `phase`: integer task-phase label, `(T,)`

`manifest.json` records seeds, initial conditions, goal, phase names, lengths,
and per-episode placement validation.  The dense observations/actions and
smooth, multimodal trajectories are suitable source data for later CVAE,
action-chunking, and temporal-ensemble conversion.

Randomly select 10 safe episodes, save a trajectory overview image, and replay
them sequentially in MuJoCo (the default behavior):

```bash
python3 visualize_trajectories.py
```

The MuJoCo overlay shows `EPISODE n / 10`, filename, frame, and speed. To use a
different playback speed:

```bash
python3 visualize_trajectories.py --replay --speed 2
```

Change `--seed` to select another reproducible random subset. Use `--show` to
open the Matplotlib figure in addition to saving it, or `--plot-only` to skip
the MuJoCo window.
