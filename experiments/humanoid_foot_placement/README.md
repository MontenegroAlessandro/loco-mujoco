# Humanoid Foot Placement

Training, evaluation, and deployment (sim2sim and sim2real) code for foot-placement
locomotion policies on the Booster T1 and Unitree G1 (27 actuated-DoF configuration)
humanoids.

All commands below are meant to be run from the **repository root** (`loco-mujoco/`),
since every config references paths (policy checkpoints, robot XMLs) relative to it.

## Directory structure

```
experiments/humanoid_foot_placement/
├── train/       # Hydra training configs + experiment.py (PPO training entry point)
├── eval/        # Offline rollout evaluation of a trained policy
├── deploy/      # MuJoCo sim2sim deployment, one script/config pair per scenario
├── flat_nav/    # Foot-placement over structured terrain (ramps, stairs, spiral
│                # staircases) + the separate velocity-tracking flat-walking policy
├── realworld/   # Real-hardware deployment bridge + camera/ArUco calibration
└── policies/    # Pretrained policy checkpoints shipped with this release
```

### Policies (`policies/`)

| File | Robot | Description |
|---|---|---|
| `T1/t1_flat.pkl` | Booster T1 | General-purpose / flat-ground policy |
| `T1/t1_steps.pkl` | Booster T1 | Foot-placement policy for structured terrain (ramps, stairs, spiral staircase) |
| `T1/t1_flat_velocity.pkl` | Booster T1 | Velocity-command flat-ground walking policy |
| `G1/g1_flat.pkl` | Unitree G1 | General-purpose / flat-ground policy |
| `G1/g1_steps.pkl` | Unitree G1 | Foot-placement policy for structured terrain |

Robot models used: `loco_mujoco/models/booster_t1/` and `loco_mujoco/models/unitree_g1/`
(G1 experiments use the 27 actuated-DoF configuration).

## Weights & Biases

Training and (optionally) evaluation log to Weights & Biases. Either run `wandb login`
once, or export `WANDB_API_KEY` in your environment, before launching `train/experiment.py`.
For `eval/eval.py`, logging is off by default (`wandb.log: false` in `eval/conf_eval.yaml`).

## Training

```bash
python experiments/humanoid_foot_placement/train/experiment.py --config-name conf_t1_steps
```

Available configs in `train/`:
- `conf_t1` — Booster T1, velocity-tracking
- `conf_t1_steps` — Booster T1, foot-placement over structured terrain
- `conf_g1` — Unitree G1

Override any field from the command line, e.g. `n_seeds=4`, or resume from a checkpoint
with `experiment.resume_from_path=<path/to/policy.pkl>`.

## Evaluation

```bash
python experiments/humanoid_foot_placement/eval/eval.py \
  experiment.agent_path=experiments/humanoid_foot_placement/policies/T1/t1_steps.pkl
```

## Sim2sim deployment (`deploy/`)

Each scenario has a matching `config_sim2sim_*.yaml`; the script name and config name
match (`sim2sim_goal_reach.py` ↔ `config_sim2sim_goal_reach.yaml`, etc.). Edit
`agent_path` in the config, or override it from the CLI:

```bash
python experiments/humanoid_foot_placement/deploy/sim2sim.py                     # generic terrain
python experiments/humanoid_foot_placement/deploy/sim2sim_goal_reach.py
python experiments/humanoid_foot_placement/deploy/sim2sim_nav.py
python experiments/humanoid_foot_placement/deploy/sim2sim_narrow_path.py
python experiments/humanoid_foot_placement/deploy/sim2sim_obstacle_avoidance.py
python experiments/humanoid_foot_placement/deploy/sim2sim_visual.py
```

For the Unitree G1, use `--config-name=config_sim2sim_g1`.

## Structured-terrain deployment & evaluation (`flat_nav/`)

Ramp, stairs, and spiral-staircase scripts share `flat_nav/fp_config.yaml` (defaults to
`policies/T1/t1_steps.pkl`):

```bash
python experiments/humanoid_foot_placement/flat_nav/sim2sim_ramp.py
python experiments/humanoid_foot_placement/flat_nav/sim2sim_stairs.py
python experiments/humanoid_foot_placement/flat_nav/sim2sim_stairs_spiral.py
python experiments/humanoid_foot_placement/flat_nav/sim2sim_lateral_ramp.py
python experiments/humanoid_foot_placement/flat_nav/sim2sim_obstacles_goal_reach.py

# quantitative batch evaluation over each terrain type
python experiments/humanoid_foot_placement/flat_nav/eval_ramp.py
python experiments/humanoid_foot_placement/flat_nav/eval_stairs.py
python experiments/humanoid_foot_placement/flat_nav/eval_spiral.py
```

`sim.py`, `new_sim.py`, and `vel_sim.py` load their config via a plain `--config` flag
instead of Hydra, so pass the config path explicitly when running from the repo root:

```bash
python experiments/humanoid_foot_placement/flat_nav/sim.py \
  --config experiments/humanoid_foot_placement/flat_nav/fp_config.yaml

python experiments/humanoid_foot_placement/flat_nav/vel_sim.py \
  --config experiments/humanoid_foot_placement/flat_nav/vel_conf.yaml   # velocity-tracking policy
```

## Real-world deployment (`realworld/`)

`realworld/0112/deploy_bridge.py` is the hardware deployment bridge for the Booster T1
(paired with `config_realworld.yaml`); `realworld/calibration/` holds the camera/ArUco
calibration utility used to set up the tracking cameras, and `restart_camera.sh`
restarts the camera service on the robot's onboard computer.
