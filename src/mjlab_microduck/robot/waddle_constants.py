import os
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg, XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "waddle" / "xmls"

WADDLE_WALK_XML: Path = _ROBOT_DIR / "waddle.xml"

assert WADDLE_WALK_XML.exists(), f"XML not found: {WADDLE_WALK_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WADDLE_WALK_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*hip_roll.*": 0.0,
        r".*left_hip_pitch.*": -0.63,
        r".*right_hip_pitch.*": 0.63,
        r".*left_knee.*": 1.37,
        r".*right_knee.*": 1.37,
        r".*left_ankle.*": -0.79,
        r".*right_ankle.*": -0.79,
        r".*neck_pitch.*": 0.0,
        r".*head_pitch.*": 0.0,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FOOT_COLLISION = CollisionCfg(
    geom_names_expr=[r"^(left|right)_foot_bottom_tpu$"],
    condim={r"^(left|right)_foot_bottom_tpu$": 3},
    priority={r"^(left|right)_foot_bottom_tpu$": 1},
    friction={r"^(left|right)_foot_bottom_tpu$": (0.6,)},
)

actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
)

WADDLE_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FOOT_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.95,
    ),
)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": WADDLE_WALK_ROBOT_CFG},
    )

    scene = Scene(scene_cfg, device="cuda:0")
    viewer.launch(scene.compile())