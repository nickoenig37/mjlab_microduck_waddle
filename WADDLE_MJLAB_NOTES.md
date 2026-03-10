# Waddle In MJLab: What Matches MicroDuck And What Does Not

This note explains how the current Waddle integration works inside this repository, what is inherited from the MicroDuck setup, and what matters if the goal is to train walking for Waddle.

The key point is that Waddle is currently set up as a velocity-tracking walking task, not an imitation task. That means the training loop does not use a polynomial motion file or a reference gait by default.

## 1. Making Sure Training Uses The GPU On This SSH Machine

### What this repo currently does

- The Python environment is CUDA-enabled.
- `torch` in this repo is currently `2.10.0+cu128`.
- `torch.cuda.is_available()` is currently `True`.
- The machine currently exposes 12 GPUs, all NVIDIA L40S.
- MJLab training selects devices through `CUDA_VISIBLE_DEVICES` and then maps the chosen visible GPU(s) to `cuda:0`, `cuda:1`, etc. internally.

Relevant files:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/pyproject.toml](CAPSTONE_Mujoco/mjlab_microduck_waddle/pyproject.toml)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/scripts/train.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/scripts/train.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/utils/gpu.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/utils/gpu.py)

### Important behavior

If you run:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

then MJLab sees only physical GPU 11, and internally trains on `cuda:0`.

That is normal.

So:
- physical GPU chosen by you: `11`
- device string inside training: `cuda:0`

Those are not contradictory.

### Best way to verify before training

Run:

```bash
CUDA_VISIBLE_DEVICES=11 uv run python -c "import os, torch; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('available=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print('current=', torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'); print('name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected result for that command:
- `CUDA_VISIBLE_DEVICES= 11`
- `count= 1`
- `current= 0`
- `name= NVIDIA L40S`

### Best practice for SSH usage

- Always set `CUDA_VISIBLE_DEVICES` explicitly when training on a shared server.
- Keep `MUJOCO_GL=egl` as MJLab already does this in the train script.
- If you ever want multi-GPU training, use MJLab's `gpu_ids` mechanism, but for a first Waddle run, use one GPU only.

### Recommendation

For now, use:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

If memory is tight or the run is unstable, lower `num-envs` before changing PPO hyperparameters.

## 2. What The `config_mjcf_*.json` Files Do And Whether Waddle Needs Them

Relevant files:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_walk.json](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_walk.json)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_ground_pick.json](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_ground_pick.json)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_standup.json](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/config_mjcf_standup.json)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/config.json](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/config.json)

### What they are

These JSON files are not training configs.

They are CAD-to-MJCF export configs used when generating MuJoCo XML from OnShape or a similar pipeline. They control things like:
- which parts become collision geometry
- which extra XML files get included
- post-processing commands to rename geoms, patch body positions, or add cameras
- which actuator class gets assigned by default

### Are they needed for training?

No, not for normal training.

Training only needs the final robot XML files plus the Python task configs.

For MicroDuck, training uses the generated XML files directly:
- `robot_walk.xml`
- `robot_standup.xml`
- `robot_ground_pick.xml`

For Waddle, training currently uses:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml)

### Does Waddle need one?

Only if you want a reproducible CAD regeneration workflow.

If you plan to keep hand-edited XML files and train from those, then you do not need to touch the JSON export config.

### Does the current Waddle layout fit this repo?

Yes, for training.

The current Waddle port is driven by:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py)

The JSON export config is not in the runtime path.

### Recommendation

- If you are not regenerating Waddle from CAD, leave the JSON alone.
- If you do regenerate from CAD later, then it is worth creating a cleaned-up Waddle export config that mirrors the MicroDuck export philosophy:
  - explicit foot collision naming
  - actuator defaults
  - optional post-import body-position patching
  - camera additions only if you need them

## 3. Sensor And Scene File Differences

Relevant files:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/sensors.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/sensors.xml)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/sensors.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/sensors.xml)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/scene.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/microduck/scene.xml)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/scene_flat_terrain.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/scene_flat_terrain.xml)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml)

### Important current behavior

The Waddle training task does not train from `scene_flat_terrain.xml`.

It currently trains from `waddle.xml` because [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py) points to that file.

That means:
- the plane terrain comes from MJLab, not your scene file
- the initial joint pose comes from `HOME_FRAME` in Python, not from the XML keyframe in `scene_flat_terrain.xml`
- the scene file is mostly a viewer/debugging convenience right now, not part of the walking training path

### So should you change `scene_flat_terrain.xml` for training?

Not as a first priority.

For the current MJLab walking task, changing `scene_flat_terrain.xml` will not change training behavior unless you also change the Python robot config to load that scene instead of `waddle.xml`.

### Sensor differences: do they matter?

Mostly no for the current Waddle walking task.

Why:
- the default walking task uses projected gravity, not raw accelerometer
- most observations come from MJLab entity state, not from your old playground-specific sensor names
- foot contact is handled through MJLab contact sensors on body subtrees, not through XML foot sensors

### One real difference that did matter

The optional raw-accelerometer path in [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/mdp.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/mdp.py) assumes MicroDuck-style sensor ordering.

I patched [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle/xmls/waddle.xml) to add MicroDuck-compatible IMU sensor aliases first in the sensor block:
- `orientation`
- `angular-velocity`
- `imu_ang_vel`
- `imu_lin_vel`
- `imu_accel`
- `root_angmom`

That makes the Waddle XML more comparable to MicroDuck and reduces future friction if you ever enable raw accelerometer observations or reuse inference code that expects MicroDuck-like IMU names.

### Recommendation

- Keep `scene_flat_terrain.xml` as a viewer/reference scene.
- Keep training on `waddle.xml` for now.
- Do not spend time tuning the scene file unless you intentionally want the scene file to become the runtime training source.

## 4. What The Tasks Folder Does, And Whether `waddle_velocity_env_cfg.py` Is The Right Walking Task

Relevant files:
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_imitation_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_imitation_env_cfg.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/mdp.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/mdp.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/reference_motion.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/reference_motion.py)
- [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/motion_loader.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/motion_loader.py)

### What the tasks folder contains

- `__init__.py`: registers task IDs with MJLab
- `microduck_velocity_env_cfg.py`: normal walking task based on velocity commands
- `microduck_imitation_env_cfg.py`: imitation-learning task using reference motions
- `microduck_standup_env_cfg.py`: stand-up recovery task
- `microduck_ground_pick_env_cfg.py`: crouch / ground-pick task
- `mdp.py`: reusable reward, observation, curriculum, and event helper functions
- `imitation_command.py`, `imitation_mdp.py`: imitation-task-specific motion logic

### Is `waddle_velocity_env_cfg.py` the right file for walking?

Yes.

If your goal is "train a walking controller from scratch with RL", then [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py) is the correct task.

It currently does these Waddle-specific changes on top of the MicroDuck walking task:
- swaps in the Waddle robot config
- changes foot contact matching to Waddle foot body names
- changes the viewer body to `trunk_assembly`
- changes action scale to `0.75`
- changes COM height reward window to `0.13` to `0.17`
- changes reset height range to `0.145` to `0.16`
- changes command ranges to Waddle-like values
- removes self-collision reward because the Waddle XML only exposes foot collision geoms right now

### Is it fully retuned for a larger, heavier robot?

No. It is a valid first-pass walking task, but it still inherits many MicroDuck assumptions.

The biggest inherited items are:
- PPO hyperparameters
- pose reward shape
- air-time reward weight
- foot clearance target
- push settings
- mass randomization percentage
- neck offset randomization behavior

So the current answer is:
- correctly structured for walking: yes
- guaranteed already optimal for a heavier robot: no

### What changed relative to your old playground repo?

In the old playground workflow, your training could be driven by a polynomial motion file or imitation-style reward shaping.

In the current Waddle MJLab walking task, none of that is used.

This task is command-conditioned RL:
- you sample a target body velocity command
- the policy outputs joint position offsets
- rewards encourage tracking that command while staying upright and moving cleanly

There is no gait template, no polynomial reference, and no pre-scripted walking trajectory in the current Waddle velocity task.

### If you want the old polynomial/reference-motion behavior

That is not the current Waddle training path.

To recreate that behavior, you would need a dedicated Waddle imitation task analogous to the MicroDuck imitation setup. That would require:
- a Waddle motion file
- a Waddle-compatible loader
- a Waddle imitation env config
- likely Waddle-specific motion reward tuning

### Recommendation

For first Waddle walking experiments, keep using the velocity task.

Only build a Waddle imitation task if your goal is specifically "learn to reproduce known gait references" rather than "learn to walk from reward and command tracking".

## 5. What Happens When You Run `uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096`

### High-level pipeline

1. `uv run train ...` launches MJLab's training CLI.
2. MJLab imports task entry points.
3. This repo registers `Mjlab-Velocity-Flat-Waddle` in [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py).
4. MJLab loads:
   - env config from `make_waddle_velocity_env_cfg()`
   - RL config from `WaddleRlCfg`
   - runner class `MicroduckOnPolicyRunner`
5. MJLab selects GPU(s) using `CUDA_VISIBLE_DEVICES`.
6. MJLab builds a `ManagerBasedRlEnv` with 4096 parallel environments.
7. The robot XML is compiled into the simulator.
8. Per environment step, the policy sees observations and outputs joint position actions.
9. Rewards, terminations, randomizations, and curricula are evaluated.
10. After rollout collection, PPO updates the policy.
11. Logs, checkpoints, and optional W&B data are written.

### In more concrete repo terms

For this task, the command path is:

- task id: `Mjlab-Velocity-Flat-Waddle`
- task registration: [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/__init__.py)
- env cfg: [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py)
- base walking logic inherited from: [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
- robot definition: [CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/src/mjlab_microduck/robot/waddle_constants.py)
- trainer script: [CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/scripts/train.py](CAPSTONE_Mujoco/mjlab_microduck_waddle/.venv/lib/python3.12/site-packages/mjlab/scripts/train.py)

### What one learning iteration means here

`WaddleRlCfg` uses:
- `num_steps_per_env = 24`

With `4096` envs, one rollout collects:

$$4096 \times 24 = 98{,}304$$

environment steps per iteration.

Then PPO trains on that batch.

### What is actually being optimized

The policy is learning joint position offsets so that Waddle:
- tracks commanded linear and angular velocity
- stays upright
- keeps a plausible walking pose
- avoids excessive slip and jerky actions
- maintains a sensible body height

It is not trying to follow a hand-authored gait trajectory in the current walking task.

## 6. What The Training Metrics Mean, And Whether Waddle Needs Different Values

### First distinction: logs vs configs

The lines you pasted are mostly logged outcomes, not input parameters.

Examples:
- `Mean reward`
- `Episode_Reward/track_linear_velocity`
- `Metrics/slip_velocity_mean`

These are measurements of training progress.

Examples of actual configured parameters are in the env and RL config files:
- reward weights
- command ranges
- PPO learning rate
- rollout length
- noise settings
- curriculum stages

### PPO / training-process metrics

#### `Learning iteration 2206/50000`
- PPO has completed 2206 training updates out of a max budget of 50000.

#### `Total steps`
- Total environment steps collected so far across all parallel envs.

#### `Steps per second`
- Throughput.
- High is better for speed, not necessarily for policy quality.

#### `Collection time`
- Time spent simulating rollouts.

#### `Learning time`
- Time spent doing PPO optimization on the collected rollout batch.

#### `Mean value loss`
- Critic error.
- Lower is usually better, but not meaningful by itself unless trends stay stable.

#### `Mean surrogate loss`
- PPO policy objective.
- The sign is not something to optimize manually from the log alone.
- Watch for stability, not a target absolute value.

#### `Mean entropy loss`
- Exploration term.
- As training settles, effective exploration usually drops.

#### `Mean reward`
- Average per-episode reward.
- Useful only relative to your own reward design.

#### `Mean episode length`
- How long episodes last before timeout or failure.
- For walking, longer often means the robot is falling less often.

#### `Mean action noise std`
- Current exploration noise standard deviation.
- This is part of policy exploration, not a robot physical parameter.

### Reward terms

These tell you where reward is coming from.

#### `Episode_Reward/track_linear_velocity`
- How well the robot matches commanded `x/y` velocity.

#### `Episode_Reward/track_angular_velocity`
- How well it matches yaw-rate commands.

#### `Episode_Reward/upright`
- Reward for keeping body orientation upright.

#### `Episode_Reward/pose`
- Reward for staying within expected joint pose ranges for standing/walking.

#### `Episode_Reward/body_ang_vel`
- Penalty for excessive body angular velocity.

#### `Episode_Reward/angular_momentum`
- Penalty for excessive total angular momentum.

#### `Episode_Reward/dof_pos_limits`
- Penalty for pushing joint motion close to limits.

#### `Episode_Reward/action_rate_l2`
- Penalty for making actions change too abruptly.

#### `Episode_Reward/air_time`
- Reward encouraging stepping cadence and non-shuffling foot behavior.

#### `Episode_Reward/foot_clearance`
- Reward or penalty related to how well feet lift during swing.

#### `Episode_Reward/foot_swing_height`
- Similar to clearance, with explicit swing height target.

#### `Episode_Reward/foot_slip`
- Penalty for foot motion while foot is supposed to be planted.

#### `Episode_Reward/soft_landing`
- Penalty for harsh impacts.

#### `Episode_Reward/self_collisions`
- Penalty for internal body collisions.
- Waddle currently does not use this reward in its walking task.

#### `Episode_Reward/stillness_at_zero_command`
- Reward for staying still when the command is near zero.

#### `Episode_Reward/neck_action_rate_l2`
- Penalty for jerky neck/head action changes.

#### `Episode_Reward/com_height_target`
- Reward for keeping the body height within a target band.

#### `Episode_Reward/joint_torques_l2`
- Penalty for large torques or aggressive effort.

### Curriculum metrics

#### `Curriculum/action_rate_weight`
- Current action-rate penalty weight from the curriculum schedule.

#### `Curriculum/standing_envs`
- Fraction of environments currently assigned to zero-command standing behavior.
- Waddle currently removes this curriculum and instead fixes `rel_standing_envs = 0.1`.

#### `Curriculum/velocity_command_ranges`
- Current command range after curriculum expansion.
- Waddle currently removes this curriculum and instead fixes the command limits.

#### `Curriculum/neck_offset_magnitude`
- Current max neck offset if that curriculum is active.

### Diagnostic metrics

#### `Metrics/twist/error_vel_xy`
- Average linear velocity tracking error.
- Lower is better.

#### `Metrics/twist/error_vel_yaw`
- Average yaw tracking error.
- Lower is better.

#### `Episode_Termination/time_out`
- Episodes ending because they survived to max length.

#### `Episode_Termination/fell_over`
- Episodes ending because the robot fell.

#### `Metrics/angular_momentum_mean`
- Average angular momentum magnitude.

#### `Metrics/air_time_mean`
- Average foot air time.

#### `Metrics/peak_height_mean`
- Average max foot swing height.

#### `Metrics/slip_velocity_mean`
- Average slipping velocity at contacts.

#### `Metrics/landing_force_mean`
- Average impact force when feet land.

### Does Waddle need different values because it is larger and heavier?

Yes for task design and reward shaping.

No for the generic PPO-loss interpretation.

In other words:
- `Mean value loss`, `Mean surrogate loss`, `Mean entropy loss` do not get changed because Waddle is heavier.
- reward scales, command ranges, reset heights, pose tolerances, contact goals, and randomization settings often do need retuning for a different robot.

### Waddle-specific settings that matter most first

The first settings most likely to matter for Waddle are:

1. Command ranges
   - already reduced for Waddle in `waddle_velocity_env_cfg.py`

2. Reset height and COM target
   - already adjusted for Waddle

3. Action scale
   - already adjusted to `0.75`

4. Pose reward tolerance
   - still inherited from MicroDuck

5. Air-time and swing-height shaping
   - still inherited from MicroDuck

6. Pushes and neck randomization
   - still inherited from MicroDuck behavior through the base env config

### Practical recommendation for first Waddle runs

Use the current Waddle task as your baseline.

If early learning is poor, the first things I would reconsider are:

1. Reduce or temporarily remove push events.
2. Reduce or temporarily remove neck offset randomization.
3. Revisit `air_time`, `foot_clearance`, and `foot_swing_height` for Waddle stride geometry.
4. Loosen or retune pose reward tolerances if Waddle needs a different natural gait envelope.

## Bottom Line

- Your current Waddle task is set up correctly enough to start walking training in MJLab.
- It is not using your old polynomial motion file. This is pure velocity-tracking RL.
- The MicroDuck `config_mjcf_*.json` files are export-time files, not training-time files.
- Your `scene_flat_terrain.xml` is currently not the training scene source, so changing it will not change training unless you also change the Python robot loader.
- The biggest remaining gap is not basic wiring; it is robot-specific reward and curriculum retuning once you observe Waddle's first real training behavior.

## Suggested First Command

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

## Suggested Verification Command

```bash
CUDA_VISIBLE_DEVICES=11 uv run python -c "import os, torch; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('available=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print('current=', torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'); print('name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```