# JDCobot200 safe zone and collision prediction

The maintained implementation uses MuJoCo FK, the five arm actuator limits,
the `graspframe` TCP, and conservative collision proxy geoms.

## Safe workspace

```bash
python safe_zone_ws_mujoco_fk.py --samples 50000
```

Outputs are written to `results/`: a PNG workspace plot, compressed sample
data, and a JSON summary. A pose is safe only when it satisfies joint limits,
self-collision clearance, and floor clearance.

## Pose or trajectory collision check

```bash
python self_collision_predictor.py --q-deg 0 0 0 0 0
python self_collision_predictor.py \
  --start-deg 0 0 0 0 0 --goal-deg 0 45 -60 30 0 --steps 201
```

Exit status is `0` for safe, `2` for unsafe, and `1` for invalid input.
Margins default to 5 mm between non-neighboring links and 10 mm above the
floor. The proxy volumes are intentionally conservative and should be tuned
against measured hardware geometry before autonomous operation.

`safe_zone_ws_DH_fk.py` is retained only as a legacy comparison; it does not
perform the maintained MuJoCo collision-clearance checks.
