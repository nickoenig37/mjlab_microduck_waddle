# Waddle Walking Training Guide (MJLab)

This guide is specific to training a walking policy for your Waddle robot in this repo.

Repository root used in examples:

```bash
cd /u50/koenin1/CAPSTONE_Mujoco/mjlab_microduck_waddle
```

## 1. Quick Start: Train Waddle Walking

Use the flat-terrain Waddle task:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

Notes:
- `Mjlab-Velocity-Flat-Waddle` is the Waddle walking task id.
- `--env.scene.num-envs` controls parallel simulation environments.
- `CUDA_VISIBLE_DEVICES=11` picks physical GPU 11 on this server.

## 2. Where Models And Logs Are Saved

Training output is saved under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

For Waddle defaults (from `waddle_velocity_env_cfg.py`):
- `experiment_name = waddle_velocity`
- `run_name = waddle_velocity`

So a typical Waddle run folder looks like:

```text
logs/rsl_rl/waddle_velocity/2026-03-10_13-00-00_waddle_velocity/
```

Inside a run folder, key files are:
- `model_0.pt`, `model_250.pt`, ...: PyTorch checkpoints saved during training.
- `events.out.tfevents.*`: TensorBoard-compatible metrics file.
- `params/env.yaml` and `params/agent.yaml`: exact config snapshot used for that run.
- `<run_folder_name>.onnx`: exported policy artifact when produced by the runner.

Important:
- The main training model checkpoint format is `.pt`.
- Resume training uses `.pt` checkpoints.
- ONNX is mainly for inference/deployment.

## 3. How To Check Results While Training Is Running

### 3.1 Watch checkpoints appear

```bash
watch -n 10 'ls -lh logs/rsl_rl/waddle_velocity/*/model_*.pt 2>/dev/null | tail -n 20'
```

### 3.2 Watch the latest run folder and files

```bash
watch -n 10 'latest=$(ls -1dt logs/rsl_rl/waddle_velocity/* 2>/dev/null | head -n 1); echo "latest=$latest"; ls -lh "$latest" | tail -n 20'
```

### 3.3 Read metrics from TensorBoard files

If TensorBoard is available:

```bash
uv run tensorboard --logdir logs/rsl_rl --port 6006
```

Then SSH port-forward from local machine:

```bash
ssh -L 6006:localhost:6006 <user>@<server>
```

Open `http://localhost:6006` locally.

### 3.4 Check terminal log file (for detached runs)

If you run with `nohup` and redirect output, monitor with:

```bash
tail -f train_waddle.log
```

## 4. Multi-GPU Training

MJLab supports multi-GPU via `--gpu-ids`.

Important behavior:
- `--gpu-ids` indexes into currently visible GPUs.
- If you set `CUDA_VISIBLE_DEVICES`, then `--gpu-ids` is relative to that filtered list.

### 4.0 What `num-envs` means when you add GPUs

- `num-envs` is the total number of parallel environments across all GPUs.
- Effective PPO rollout batch per iteration is:
  - `num-envs * agent.num_steps_per_env`

With the default `num_steps_per_env = 24`:
- `num-envs=4096` gives `98,304` samples per iteration.
- `num-envs=8192` gives `196,608` samples per iteration.

Why people increase `num-envs` when adding GPUs:
- If you keep `num-envs` fixed and add GPUs, each GPU gets fewer envs and you may not use the extra hardware efficiently.
- Increasing `num-envs` raises simulator throughput and usually improves wall-clock training speed.

Important: adding GPUs is not always exactly the same training run, just faster.
- More GPUs by itself can be "same run, faster" if `num-envs` is unchanged.
- But in practice, you often increase `num-envs`; that changes the per-iteration batch size and can change learning dynamics.
- Bigger batches tend to be less noisy and may need minor tuning later (for example learning rate, epochs, or reward weights), even if behavior is often similar.

### 4.0.1 Why increase gradually

Increasing `num-envs` too aggressively can cause:
- GPU OOM
- slower iterations from communication overhead
- less stable early learning because optimization dynamics shifted too much at once

Use a ramp, check training curves, then increase again.

Suggested ramp from your current baseline:
- 1 GPU: `4096`
- 2 GPUs: `6144`
- 4 GPUs: `8192`
- 7 GPUs: `12288`

These are starting points, not strict rules.

Examples:

### 4.1 Use all currently visible GPUs

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids all --env.scene.num-envs 8192
```

### 4.2 Use two GPUs from a visible set

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids 0 1 --env.scene.num-envs 8192
```

In this example, `--gpu-ids 0 1` maps to physical GPUs `8` and `9`.

### 4.3 Use specific physical GPUs directly

```bash
CUDA_VISIBLE_DEVICES=2,5 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids all --env.scene.num-envs 8192
```

This uses physical GPUs 2 and 5.

### 4.4 Four-GPU example

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids all --env.scene.num-envs 8192
```

### 4.5 Seven-GPU example

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids all --env.scene.num-envs 12288
```

If 7-GPU run is stable and under-utilized, try `14336` next.
If unstable or memory-limited, drop to `10240`.

Practical advice:
- Increase `num-envs` when adding GPUs, but do it gradually.
- If you hit OOM, reduce `num-envs` first.
- Keep one run per GPU set to avoid contention.

### More advice:
- Use this first when going back to trying multiple:
```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run train Mjlab-Velocity-Flat-Waddle \
  --gpu-ids all \
  --env.scene.num-envs 8192 \
  --agent.save-interval 50
```

- To monitor outputs in the same way that 1 GPU might do it:
```bash
run=$(ls -1dt logs/rsl_rl/waddle_velocity/* | head -n 1)
log=$(find "$run/torchrunx" -type f -name 'localhost[[]0[]].log' | head -n 1)
tail -f "$log"
```

- Estimating Completion time:
```bash
grep -E 'Learning iteration [0-9]+/[0-9]+' "$log" | tail -n 5
```

## 5. Resume Training From A Partial Checkpoint

Resume uses these flags:
- `--agent.resume True`
- `--agent.load-run <run_folder_name>`
- `--agent.load-checkpoint <checkpoint_filename>`

Example:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run 2026-03-10_13-00-00_waddle_velocity \
  --agent.load-checkpoint model_2000.pt
```

How to find latest checkpoint in a run:

```bash
run=logs/rsl_rl/waddle_velocity/2026-03-10_13-00-00_waddle_velocity
ls -1 "$run"/model_*.pt | sort -V | tail -n 1
```

Important:
- Keep the same task id when resuming.
- Keep core architecture settings unchanged when resuming (policy dimensions, etc.).

## 6. Run Training On SSH And Disconnect Safely

Use either `tmux` (preferred) or `nohup`.

### 6.1 Method A: tmux (recommended)

Start session:

```bash
tmux new -s waddle_train
```

Run training command inside tmux:

```bash
cd /u50/koenin1/CAPSTONE_Mujoco/mjlab_microduck_waddle
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

Detach from tmux:
- Press `Ctrl+b`, then `d`.

Reattach later:

```bash
tmux attach -t waddle_train
```

### 6.2 Method B: nohup

```bash
cd /u50/koenin1/CAPSTONE_Mujoco/mjlab_microduck_waddle
nohup bash -lc 'CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096' > train_waddle.log 2>&1 &
```

Check process:

```bash
ps -ef | grep 'uv run train Mjlab-Velocity-Flat-Waddle' | grep -v grep
```

Watch log:

```bash
tail -f train_waddle.log
```

## 7. Run Simulation To Test A Trained Model

Use MJLab `play` with a checkpoint.

Example:

```bash
uv run play Mjlab-Velocity-Flat-Waddle \
  --agent trained \
  --checkpoint-file logs/rsl_rl/waddle_velocity/2026-03-10_13-00-00_waddle_velocity/model_2000.pt \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native
```

If you only want a quick sanity check:

```bash
uv run play Mjlab-Velocity-Flat-Waddle --agent trained --checkpoint-file <path_to_model.pt> --num-envs 1
```

Optional video capture during play:

```bash
uv run play Mjlab-Velocity-Flat-Waddle --agent trained --checkpoint-file <path_to_model.pt> --video True --video-length 600
```

## 8. Parameters You Will Most Likely Tune First

These are the highest-impact knobs for Waddle walking quality and stability.

### 8.1 Runtime scale and throughput
- `--env.scene.num-envs`
- `--gpu-ids`

Why tune:
- Improves wall-clock speed and sample throughput.
- Too high can cause GPU OOM or instability.

### 8.2 Command ranges (how hard commands are)
From `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`:
- `command.ranges.lin_vel_x`
- `command.ranges.lin_vel_y`
- `command.ranges.ang_vel_z`

Why tune:
- Too wide early can prevent gait formation.
- Start conservative, widen later.

### 8.3 Action scaling
From `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`:
- `joint_pos_action.scale`

Why tune:
- Higher = stronger joint movement, can become jerky.
- Lower = safer but can under-actuate.

### 8.4 Height targets and reset pose band
From `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`:
- `com_height_target` min/max
- `reset_base` z range

Why tune:
- Must match Waddle geometry and stable walking posture.

### 8.5 Foot motion shaping
Found in `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`:
* Inherited from MicroDuck base config for `microduck_velocity_env_cfg.py`:
- `foot_clearance` weight and target height
- `foot_swing_height` weight and target height
- `air_time` weight and thresholds
- `foot_slip` penalty

Why tune:
- Directly affects stepping style, dragging, and slip behavior.

### 8.6 Tracking vs smoothness tradeoff
In base config:
- `track_linear_velocity` weight/std
- `track_angular_velocity` weight/std
- `action_rate_l2` weight
- `joint_torques_l2` weight

Why tune:
- Better command tracking often increases aggressive motion.
- Smoothness penalties reduce jitter but can make robot sluggish.

### 8.7 Domain randomization and pushes
Main toggles in `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`:
- `ENABLE_VELOCITY_PUSHES`
- `ENABLE_COM_RANDOMIZATION`
- `ENABLE_MASS_INERTIA_RANDOMIZATION`
- `ENABLE_NECK_OFFSET_RANDOMIZATION`
- IMU/base orientation randomization toggles

Why tune:
- Helps robustness, but can slow or destabilize early learning.
- Common strategy: reduce randomization for initial gait learning, re-enable later.

### 8.8 PPO learning config
From `WaddleRlCfg` in `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`:
- `learning_rate`
- `num_learning_epochs`
- `num_mini_batches`
- `num_steps_per_env`
- `max_iterations`
- `save_interval`
- `entropy_coef`
- `desired_kl`

Why tune:
- Controls optimization stability and speed.
- Tune after task/reward shaping, not before.

## 9. How And Where To Change Training Parameters

You have two ways.

### 9.1 Permanent defaults in code
Edit these files:
- `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
- `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`

Use this for:
- task defaults you want to keep across runs.

### 9.2 Per-run CLI overrides (no code edit)
Examples:

```bash
uv run train Mjlab-Velocity-Flat-Waddle \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.save-interval 100
```

```bash
uv run train Mjlab-Velocity-Flat-Waddle \
  --env.commands.twist.ranges.lin-vel-x -0.10 0.10 \
  --env.commands.twist.ranges.ang-vel-z -0.8 0.8
```

```bash
uv run train Mjlab-Velocity-Flat-Waddle \
  --env.rewards.track-linear-velocity.weight 4.0 \
  --env.rewards.action-rate-l2.weight -0.8
```

Tip:
- Prefer CLI overrides for experiments.
- Move proven settings back into code defaults later.

## 10. What The Main Training Parameters Mean

Core practical meanings:
- `num-envs`: Number of parallel worlds simulated each step.
- `num_steps_per_env`: Rollout horizon before each PPO update.
- `max_iterations`: Number of PPO update iterations.
- `save_interval`: Save checkpoint every N iterations.
- `learning_rate`: Optimizer step size.
- `num_learning_epochs`: PPO passes over each rollout.
- `num_mini_batches`: Split rollout batch into this many chunks per epoch.
- `entropy_coef`: Exploration pressure.
- `desired_kl`: Target KL for adaptive step-size behavior.
- `track_*` reward weights/std: Command-following strength and tolerance.
- `action_rate_l2`: Penalizes abrupt action changes.
- `foot_*` rewards: Shape stepping pattern and foot behavior.
- `com_height_target`: Encourages body height range.

## 11. Known Good Baseline Command Set

Single GPU baseline:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle --env.scene.num-envs 4096
```

2-GPU baseline (example):

```bash
CUDA_VISIBLE_DEVICES=10,11 uv run train Mjlab-Velocity-Flat-Waddle --gpu-ids all --env.scene.num-envs 8192
```

Resume baseline:

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <run_folder_name> \
  --agent.load-checkpoint <model_x.pt>
```

Play baseline:

```bash
uv run play Mjlab-Velocity-Flat-Waddle --agent trained --checkpoint-file <path_to_model.pt> --num-envs 1 --viewer native
```

## 12. Scaling From Your `medium.json` (Placo) To Waddle RL

You shared these key values from your previous walking config:
- `walk_com_height = 0.205`
- `walk_foot_height = 0.035`
- `walk_max_dx_forward = 0.08`
- `walk_max_dx_backward = 0.03`
- `walk_max_dy = 0.1`
- `walk_max_dtheta = 1.0`
- `single_support_duration = 0.15`
- `double_support_ratio = 0.60`

Not all map 1:1 into MJLab RL, but several do map well.

### 12.1 What maps cleanly

- COM height target range:
  - current Waddle default: `0.13` to `0.17`
  - placo reference center: around `0.205`

- Foot lift targets:
  - current RL foot target height: `0.02`
  - placo reference foot height: `0.035`

- Command limits:
  - current Waddle default: `lin_x ±0.15`, `lin_y ±0.2`, `yaw ±1.0`
  - placo reference intent: forward `+0.08`, backward `-0.03`, lateral `±0.1`, yaw `±1.0`

### 12.2 Suggested Waddle starter values (scaled toward your old gait)

Use this as a first experiment set:
- `lin_vel_x`: `(-0.06, 0.10)`
- `lin_vel_y`: `(-0.10, 0.10)`
- `ang_vel_z`: `(-1.0, 1.0)`
- `foot_clearance.target_height`: `0.03`
- `foot_swing_height.target_height`: `0.03`
- `com_height_target`: `0.16` to `0.19`

Why this set:
- It preserves your prior "effect" (moderate forward speed, smaller backward speed behavior, meaningful foot lift, higher torso than MicroDuck defaults).
- It avoids jumping directly to a strict `0.205` COM target, which may be too aggressive for initial RL stabilization.

These values are now applied as the default Waddle task settings in:
- `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`

### 12.3 CLI override example for this scaled set

```bash
CUDA_VISIBLE_DEVICES=11 uv run train Mjlab-Velocity-Flat-Waddle \
  --env.scene.num-envs 4096 \
  --env.commands.twist.ranges.lin-vel-x -0.06 0.10 \
  --env.commands.twist.ranges.lin-vel-y -0.10 0.10 \
  --env.commands.twist.ranges.ang-vel-z -1.0 1.0 \
  --env.rewards.foot-clearance.params.target-height 0.03 \
  --env.rewards.foot-swing-height.params.target-height 0.03 \
  --env.rewards.com-height-target.params.target-height-min 0.16 \
  --env.rewards.com-height-target.params.target-height-max 0.19
```

### 12.4 Optional second stage (if stage 1 is stable)

If walking is stable and not too jittery after stage 1, try:
- `foot_* target_height`: `0.032` to `0.035`
- `com_height_target`: `0.17` to `0.20`

Move in small increments and keep one change set per run so results are attributable.

### 12.5 Notes on items that do not map directly

- `double_support_ratio`, `single_support_duration`, and ZMP terms are planner-phase concepts.
- In RL, closest equivalents are reward shaping terms (`air_time`, `foot_swing_height`, `foot_clearance`, `foot_slip`) and command ranges.
- So treat those planner values as intent targets, not direct 1:1 parameters.

### 12.6 Exact "where to change" mapping

If you want to edit these defaults later, use this map.

- Forward/backward command range:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `command.ranges.lin_vel_x`

- Lateral command range:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `command.ranges.lin_vel_y`

- Yaw command range:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `command.ranges.ang_vel_z`

- COM height target:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - keys: `cfg.rewards["com_height_target"].params["target_height_min"]`, `cfg.rewards["com_height_target"].params["target_height_max"]`

- Foot clearance target:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `cfg.rewards["foot_clearance"].params["target_height"]`

- Foot swing height target:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `cfg.rewards["foot_swing_height"].params["target_height"]`

- Action scale:
  - file: `src/mjlab_microduck/tasks/waddle_velocity_env_cfg.py`
  - key: `joint_pos_action.scale`

- Values inherited from MicroDuck base walking config:
  - file: `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
  - keys: `air_time`, `track_linear_velocity`, `track_angular_velocity`, `action_rate_l2`, `foot_slip`, `joint_torques_l2`, randomization toggles, and curricula definitions

Use `waddle_velocity_env_cfg.py` for Waddle-specific defaults first.
Only change `microduck_velocity_env_cfg.py` when you intentionally want to change inherited shared behavior.
