"""Waddle velocity environment."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.robot.waddle_constants import WADDLE_WALK_ROBOT_CFG

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg


def make_waddle_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a Waddle velocity tracking environment configuration."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    site_names = ["left_foot", "right_foot"]

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot_assembly|foot_assembly_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    cfg.scene.entities = {"robot": WADDLE_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg,)
    cfg.viewer.body_name = "trunk_assembly"

    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.75

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_assembly",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_assembly",)
    # Raised to better match prior Waddle walking targets while staying conservative.
    cfg.rewards["com_height_target"].params["target_height_min"] = 0.165
    cfg.rewards["com_height_target"].params["target_height_max"] = 0.22

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Increase swing/clearance targets toward prior planner foot height.
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    if "self_collisions" in cfg.rewards:
        del cfg.rewards["self_collisions"]

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_bottom_tpu",
        "right_foot_bottom_tpu",
    )
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.145, 0.16)

    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["asset_cfg"].body_names = ("trunk_assembly",)

    if "randomize_mass_inertia" in cfg.events:
        cfg.events["randomize_mass_inertia"].params["asset_cfg"].body_names = (
            "trunk_assembly",
        )

    # In play mode, remove external push perturbations so behavior reflects
    # policy quality rather than frequent scripted disturbances.
    if play and "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.1
    # Bias toward forward walking with smaller reverse/lateral limits.
    command.ranges.lin_vel_x = (-0.06, 0.10)
    command.ranges.lin_vel_y = (-0.10, 0.10)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.viz.z_offset = 0.6

    if "standing_envs" in cfg.curriculum:
        del cfg.curriculum["standing_envs"]
    if "velocity_command_ranges" in cfg.curriculum:
        del cfg.curriculum["velocity_command_ranges"]

    return cfg


WaddleRlCfg = deepcopy(MicroduckRlCfg)
WaddleRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="waddle_velocity",
    run_name="waddle_velocity",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)