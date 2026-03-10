# MJLab MicroDuck — Complete Usage Guide

This guide covers how to set up, train, evaluate, and deploy RL policies for the MicroDuck robot using the MJLab framework.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Setup & Installation](#2-setup--installation)
3. [Repository Structure](#3-repository-structure)
4. [Available Tasks](#4-available-tasks)
5. [Training](#5-training)
6. [Playing / Evaluating Trained Policies](#6-playing--evaluating-trained-policies)
7. [Exporting to ONNX](#7-exporting-to-onnx)
8. [Running ONNX Inference (scripts/infer_policy.py)](#8-running-onnx-inference)
9. [Replaying Reference Motions](#9-replaying-reference-motions)
10. [Plotting Observation Comparisons](#10-plotting-observation-comparisons)
11. [Robot Configuration & MJCF Model](#11-robot-configuration--mjcf-model)
12. [Task Details & Reward Structure](#12-task-details--reward-structure)
13. [Weights & Biases Integration](#13-weights--biases-integration)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

This repository trains RL policies for **MicroDuck**, a 14-DOF bipedal robot, using the [MJLab](https://github.com/mujocolab/mjlab) framework (built on MuJoCo + RSL-RL + PyTorch).

**Four distinct tasks** are implemented:

| Task | Description |
|------|-------------|
| **Velocity Tracking** | Walk at commanded velocities (forward/backward/lateral/turning) |
| **Stand-Up** | Recover from lying on its back to standing |
| **Ground Pick** | Crouch down, touch the ground with its mouth, return to standing |
| **Imitation** | Track velocity-indexed reference motions from polynomial-fitted trajectories |

Each task has flat and rough terrain variants (except imitation which is flat-only).

---

## 2. Setup & Installation

### Prerequisites

- **Python 3.12+** (required by the project, `uv` will handle this automatically)
- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **NVIDIA GPU with CUDA** for training (inference works on CPU)
- **Weights & Biases account** for experiment tracking (optional but recommended)

### Quick Setup

```bash
# Navigate to the repo
cd mjlab_microduck_waddle

# Create venv and install all dependencies
# uv handles Python 3.12 automatically
# CUDA PyTorch is configured in pyproject.toml — no manual steps needed
uv venv --python 3.12 .venv
uv sync

# Install extra script dependencies (for inference, plotting, keyboard control)
uv pip install pynput onnxruntime plotly
```

That's it. The `pyproject.toml` is configured to pull PyTorch from the CUDA 12.8 index automatically, so `uv sync` and `uv run` will always install the GPU-enabled version.

> **Different CUDA version?** If your GPU needs a different CUDA version (check with `nvidia-smi`), edit the `[[tool.uv.index]]` URL in `pyproject.toml` — change `cu128` to `cu118`, `cu121`, or `cu124`, then run `uv lock --upgrade-package torch --upgrade-package torchvision` followed by `uv sync`.

### Verify Installation

```bash
uv run python -c "
import torch, mujoco, mjlab, mjlab_microduck
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MuJoCo: {mujoco.__version__}')
print('All OK!')
"
```

You should see all tasks register:
```
✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck
✓ Ground pick task registered: Mjlab-GroundPick-Flat-MicroDuck
✓ Imitation task registered: Mjlab-Imitation-Flat-MicroDuck
```

---

## 3. Repository Structure

```
mjlab_microduck_waddle/
├── pyproject.toml              # Project config, dependencies, task entry point
├── export.py                   # ONNX export script
├── scripts/
│   ├── infer_policy.py         # Run ONNX policy in MuJoCo with keyboard control
│   ├── replay_reference_motion.py  # Replay reference motions for verification
│   └── plot_observations_comparison_plotly.py  # Compare real vs sim observations
├── src/mjlab_microduck/
│   ├── __init__.py
│   ├── motion_loader.py        # Frame-based velocity-indexed motion loader
│   ├── reference_motion.py     # Polynomial-fitted reference motion evaluator
│   ├── data/
│   │   └── reference_motion.pkl  # Pre-computed reference motions for imitation
│   ├── robot/
│   │   ├── microduck_constants.py  # Joint config, home pose, actuator params
│   │   └── microduck/             # MJCF model files
│   │       ├── scene.xml          # Main simulation scene
│   │       ├── robot_walk.xml     # Walking variant (foot-only collision)
│   │       ├── robot_standup.xml  # Stand-up variant (full body collision)
│   │       ├── robot_ground_pick.xml  # Ground pick variant
│   │       ├── sensors.xml        # IMU, gyro, accelerometer definitions
│   │       ├── joints_properties.xml  # Actuator gain/force limit classes
│   │       ├── config_mjcf_walk.json  # OnShape export config
│   │       └── assets/            # STL mesh files
│   └── tasks/
│       ├── __init__.py            # Task registration (entry point)
│       ├── mdp.py                 # Common MDP functions (rewards, actions)
│       ├── imitation_command.py   # Imitation command manager
│       ├── imitation_mdp.py       # Imitation-specific observations/rewards
│       ├── microduck_velocity_env_cfg.py   # Velocity task config (~600 lines)
│       ├── microduck_imitation_env_cfg.py  # Imitation task config (~670 lines)
│       ├── microduck_standup_env_cfg.py    # Stand-up task config
│       └── microduck_ground_pick_env_cfg.py # Ground pick task config
```

---

## 4. Available Tasks

All tasks are registered via the plugin entry point `mjlab.tasks` in `pyproject.toml`.

| Task ID | Terrain | Description |
|---------|---------|-------------|
| `Mjlab-Velocity-Flat-MicroDuck` | Flat | Velocity tracking on flat ground |
| `Mjlab-Velocity-Rough-MicroDuck` | Rough | Velocity tracking on rough terrain |
| `Mjlab-StandUp-Flat-MicroDuck` | Flat | Stand-up from inverted position |
| `Mjlab-StandUp-Rough-MicroDuck` | Rough | Stand-up on rough terrain |
| `Mjlab-GroundPick-Flat-MicroDuck` | Flat | Crouch & touch ground with mouth |
| `Mjlab-GroundPick-Rough-MicroDuck` | Rough | Ground pick on rough terrain |
| `Mjlab-Imitation-Flat-MicroDuck` | Flat | Track velocity-indexed reference motions |

List all available tasks:
```bash
uv run train --help
```

---

## 5. Training

### Basic Training

```bash
# Velocity tracking (most common starting point)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# Stand-up task
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096

# Ground pick task
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096

# Imitation motion tracking
uv run train Mjlab-Imitation-Flat-MicroDuck --env.scene.num-envs 4096

# Rough terrain variants
uv run train Mjlab-Velocity-Rough-MicroDuck --env.scene.num-envs 4096
```

> **First run:** The first training launch will take several minutes to compile Warp CUDA kernels. These are cached for subsequent runs.

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--env.scene.num-envs` | Number of parallel environments | Task-dependent |
| `--agent.run-name` | W&B run name | Auto-generated |
| `--agent.max-iterations` | Max training iterations | 30000 |

### Resume Training

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.run-name resume \
  --agent.load-checkpoint model_29999.pt \
  --agent.resume True
```

> **Note:** The custom `MicroduckOnPolicyRunner` automatically syncs `common_step_counter` on resume, so curriculum-based schedules (command ranges, reward weights, standing environment fraction) resume at the correct stage rather than resetting.

### Training Tips

- **4096 environments** is a good default for an RTX 4070 Laptop GPU
- Velocity task typically converges in ~5000-10000 iterations
- Imitation task may need more iterations due to the larger action space of motion tracking
- Watch W&B dashboards for reward curves and curriculum progression
- Training logs are saved under `logs/rsl_rl/<experiment_name>/`

---

## 6. Playing / Evaluating Trained Policies

### Play from W&B

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <user/project/run_id>
```

### Play with Ghost Visualization (Imitation only)

```bash
# Show ghost robot tracking the reference motion
GHOST=1 uv run play Mjlab-Imitation-Flat-MicroDuck --wandb-run-path <...>

# Or use --ghost flag
uv run play Mjlab-Imitation-Flat-MicroDuck --wandb-run-path <...> --ghost
```

### Play from Local Checkpoint

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <...> \
  --checkpoint 15000
```

---

## 7. Exporting to ONNX

Export a trained policy to ONNX format for deployment on the physical robot.

```bash
# Export from W&B run
uv run export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <user/project/run_id> \
  --onnx-file my_policy.onnx

# Export specific checkpoint
uv run export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <...> \
  --checkpoint 15000 \
  --onnx-file velocity_15k.onnx

# Export from local checkpoint file
uv run export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/.../model_15000.pt \
  --onnx-file velocity_local.onnx

# Export imitation policy (automatically embeds gait period in ONNX metadata)
uv run export.py Mjlab-Imitation-Flat-MicroDuck \
  --wandb-run-path <...> \
  --onnx-file imitation_policy.onnx
```

### Export Options

| Option | Description |
|--------|-------------|
| `--onnx-file` | Output ONNX file path (default: `output.onnx`) |
| `--wandb-run-path` | W&B run path (`user/project/run_id`) |
| `--checkpoint` | Specific checkpoint iteration number |
| `--checkpoint-file` | Direct path to `.pt` checkpoint file |
| `--motion-file` | Override reference motion file (imitation tasks) |
| `--agent` | Agent type: `trained`, `zero`, `random` |

For imitation policies, the gait period is automatically extracted from the reference motion and embedded in the ONNX model metadata.

---

## 8. Running ONNX Inference

The `scripts/infer_policy.py` script runs exported ONNX policies in MuJoCo with a real-time viewer and keyboard control.

> **Important:** Run this from the repo root directory (it expects `src/mjlab_microduck/robot/microduck/scene.xml` as a relative path).

### Basic Usage

```bash
# Run a walking policy
uv run python scripts/infer_policy.py --walking my_policy.onnx

# Run with both walking and standing policies (auto-switches by velocity)
uv run python scripts/infer_policy.py \
  --walking walk_policy.onnx \
  --standing stand_policy.onnx

# Run imitation policy (requires --imitation flag + reference motion for gait period)
uv run python scripts/infer_policy.py \
  --walking imitation_policy.onnx \
  --imitation \
  --reference-motion src/mjlab_microduck/data/reference_motion.pkl

# Add actuator delay simulation (MIN MAX timesteps)
uv run python scripts/infer_policy.py \
  --walking my_policy.onnx \
  --delay 1 3

# With ground pick policy (press G to trigger)
uv run python scripts/infer_policy.py \
  --walking walk.onnx \
  --standing stand.onnx \
  --ground-pick ground_pick.onnx
```

### Keyboard Controls

| Mode | Key | Action |
|------|-----|--------|
| **Velocity (default)** | UP/DOWN | Forward/backward (±0.5 m/s) |
| | LEFT/RIGHT | Lateral movement (±0.5 m/s) |
| | A / E | Turn left/right (±4.0 rad/s) |
| | SPACE | Stop (zero velocity) |
| | G | Trigger ground pick cycle |
| | B | Toggle body pose mode |
| | H | Toggle head control mode |
| **Body Pose (B)** | UP/DOWN | Height ±1mm (max ±30mm) |
| | LEFT/RIGHT | Pitch ±1° (max ±30°) |
| | A / E | Roll ±1° (max ±30°) |
| | SPACE | Reset to zero |
| **Head (H)** | Z / S | Neck pitch ±0.1 rad |
| | UP/DOWN | Head pitch ±0.1 rad |
| | LEFT/RIGHT | Head yaw ±0.1 rad |
| | A / E | Head roll ±0.1 rad |
| | SPACE | Reset head to zero |

### Policy Switching

When both `--walking` and `--standing` are provided, the script auto-switches:
- **Walking policy** when velocity command magnitude > threshold (default 0.05)
- **Standing policy** when velocity command magnitude ≤ threshold
- Adjust threshold with `--switch-threshold 0.1`

### Inference Script Options

| Option | Description |
|--------|-------------|
| `--walking PATH` | Walking policy ONNX file |
| `--standing PATH` | Standing policy ONNX file |
| `--ground-pick PATH` | Ground pick policy ONNX file |
| `--imitation` | Enable imitation mode (adds phase observations) |
| `--reference-motion PATH` | Reference motion `.pkl` for gait period |
| `--action-scale FLOAT` | Action scaling factor (default: 1.0) |
| `--delay [MIN MAX]` | Actuator delay in timesteps |
| `--raw-accelerometer` | Use raw accelerometer instead of projected gravity |
| `--debug` | Print observations and actions |
| `--save-csv PATH` | Save observations/actions to CSV |
| `--record PATH` | Record observations to pickle file |
| `--switch-threshold FLOAT` | Walking/standing switch threshold (default: 0.05) |
| `--ground-pick-period FLOAT` | Ground pick cycle duration in seconds (default: 4.0) |

### Observation Format

**Velocity/Standing (51D):**
`[base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(3)]`

**Imitation (53D):**
`[command(3), phase(2), base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14)]`

Joint positions are relative to the default (home) pose. The default pose is:
```
left_hip_yaw=0, left_hip_roll=0, left_hip_pitch=0.6, left_knee=-1.2, left_ankle=0.6,
neck_pitch=-0.5, head_pitch=0.5, head_yaw=0, head_roll=0,
right_hip_yaw=0, right_hip_roll=0, right_hip_pitch=-0.6, right_knee=1.2, right_ankle=-0.6
```

---

## 9. Replaying Reference Motions

Verify reference motion correctness by playing them back in MuJoCo.

```bash
# Replay with default (first) motion
uv run python scripts/replay_reference_motion.py \
  --reference-motion src/mjlab_microduck/data/reference_motion.pkl

# Replay a specific motion by key (e.g., "0.1_0.0_0.0")
uv run python scripts/replay_reference_motion.py \
  --reference-motion src/mjlab_microduck/data/reference_motion.pkl \
  --motion-key "0.1_0.0_0.0"
```

### Keyboard Shortcuts (in replay viewer)

Motions are auto-categorized by velocity direction:
- **Numeric keys** or **arrow-like patterns** to switch between forward/backward/left/right/rotate motions
- The script prints available motions and their categories on startup

The replay also computes and displays imitation reward components with error breakdowns, useful for debugging reward design.

---

## 10. Plotting Observation Comparisons

Compare real robot observations vs simulated observations using interactive Plotly charts.

```bash
uv run python scripts/plot_observations_comparison_plotly.py \
  --real-csv real_obs.csv \
  --sim-csv sim_obs.csv
```

This generates a 4-column subplot with:
- Command values
- Phase (if imitation)
- Base angular velocity
- Projected gravity
- Joint positions
- Joint velocities
- Actions

Supports both velocity (51D) and imitation (53D) observation formats.

---

## 11. Robot Configuration & MJCF Model

### Joint Configuration (14 DOF)

| # | Joint | Range |
|---|-------|-------|
| 0 | `left_hip_yaw` | — |
| 1 | `left_hip_roll` | — |
| 2 | `left_hip_pitch` | — |
| 3 | `left_knee` | — |
| 4 | `left_ankle` | — |
| 5 | `neck_pitch` | — |
| 6 | `head_pitch` | — |
| 7 | `head_yaw` | — |
| 8 | `head_roll` | — |
| 9 | `right_hip_yaw` | — |
| 10 | `right_hip_roll` | — |
| 11 | `right_hip_pitch` | — |
| 12 | `right_knee` | — |
| 13 | `right_ankle` | — |

### Actuator Properties

| Property | Value |
|----------|-------|
| Kp (position gain) | 0.52 |
| Kv (velocity gain) | 0.0 |
| Force limit | ±0.91 N |
| Joint damping | 0.048 |
| Joint friction | 0.006 |

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (control at 50 Hz) |
| Control frequency | 50 Hz |
| Physics frequency | 200 Hz |

### Model Variants

- **`robot_walk.xml`** — Foot-only collisions (for walking tasks)
- **`robot_standup.xml`** — Full body collisions including shells and head (for stand-up)
- **`robot_ground_pick.xml`** — Selective collisions (foot terrain + body self-collision only)

### Sensors (defined in `sensors.xml`)

| Sensor | Type | Noise |
|--------|------|-------|
| `imu_frame_quat` | Framequat | 0.001 |
| `gyro` | Gyroscope | 0.005 |
| `imu_ang_vel` | Gyroscope | None |
| `imu_lin_vel` | Velocimeter | None |
| `imu_accel` | Accelerometer | None |

---

## 12. Task Details & Reward Structure

### Velocity Tracking Task

The core walking task. The robot receives velocity commands `[lin_vel_x, lin_vel_y, ang_vel_z]` and must track them.

**Key features:**
- Extensive domain randomization (COM offset, mass/inertia scaling, PD gains, joint friction/damping, IMU orientation)
- Curriculum-based command range expansion
- Standing environment fraction (subset of envs where robot must stand still)
- Neck offset randomization for robustness

**Reward components include:**
- Velocity tracking (linear + angular)
- Joint acceleration penalty
- Foot air time
- Foot clearance
- Action rate smoothness
- Pose regularization

### Imitation Task

Builds on the velocity task but replaces commands/rewards with motion-tracking objectives.

**Key features:**
- Velocity-indexed motion library loaded from `reference_motion.pkl`
- Polynomial-fitted trajectories evaluated at arbitrary phases
- Adaptive sampling that weights difficult velocities higher
- Ghost visualization for debugging (set `GHOST=1` or `--ghost`)

**Imitation reward components (BD-X paper structure):**
- Joint position tracking
- Joint velocity tracking
- Base velocity tracking
- Contact pattern matching
- Foot clearance matching
- Phase consistency

### Stand-Up Task

Robot starts inverted (lying on back) and must right itself.

**Key features:**
- Body pose command control (height + pitch/roll)
- Curriculum-based weight ramps
- Upright reward + COM upward velocity reward
- Phase-dependent pose tracking

### Ground Pick Task

Episodic task: crouch, touch ground with mouth, return to standing.

**Key features:**
- Phase-encoded commands (cos/sin of phase)
- Mouth proximity and perpendicularity rewards
- Loose leg tracking + tight neck tracking
- Automatic phase progression

---

## 13. Weights & Biases Integration

Training automatically logs to W&B. Set up:

```bash
# Login to W&B (one-time)
uv run wandb login

# Training will auto-log to W&B
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

W&B is used for:
- **Experiment tracking** — reward curves, curriculum progress, training metrics
- **Checkpoint storage** — models are uploaded as artifacts
- **Motion artifact storage** — reference motions for imitation tasks
- **Run comparison** — compare different hyperparameter configurations

The `--wandb-run-path` argument (format: `user/project/run_id`) is used in `play` and `export.py` to download checkpoints and motion files from W&B.

---

## 14. Troubleshooting

### Common Issues

**`uv sync` installs CPU-only PyTorch**
This should not happen — the `pyproject.toml` is configured to pull CUDA PyTorch automatically. If it does, regenerate the lockfile:
```bash
uv lock --upgrade-package torch --upgrade-package torchvision
uv sync
```

**"Imitation task not registered" warning**
The file `src/mjlab_microduck/data/reference_motion.pkl` must exist. If it's missing, the imitation task won't register. This file should be included in the repo (check with `git lfs pull` if using LFS).

**ONNX inference observation size mismatch**
Ensure you use the correct flags:
- Velocity/Standing policies: no extra flags needed (51D observations)
- Imitation policies: add `--imitation` flag (53D observations)
- If using raw accelerometer: add `--raw-accelerometer`

**MuJoCo viewer doesn't open**
Ensure you have a display available. On headless servers, use `--video` flag with `play` command to record videos instead.

**Keyboard controls not working in infer_policy.py**
Install `pynput`: `uv pip install pynput`. On some systems, keyboard capture requires elevated privileges.

**`uv run` takes a long time on first run**
The first `uv run` compiles the project. Subsequent runs are much faster.

**Warp deprecation warnings**
These are harmless warnings from the `warp-lang` dependency and can be ignored.

### Getting Help

- MJLab framework: https://github.com/mujocolab/mjlab
- MicroDuck hardware: https://github.com/apirrone/microduck
- MuJoCo docs: https://mujoco.readthedocs.io/
