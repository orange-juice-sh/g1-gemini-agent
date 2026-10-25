"""しゃがむタスクの環境設定（体・PD・制御周期は MuJoCo 側とそろえる）"""
from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp  # noqa: E402

from . import mdp as squat

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/g1_description/g1_29dof_lock_waist.urdf")

from .spec import ACTION_SCALES, ARMS, DEFAULT, LEGS, POLICY_JOINTS  # noqa: E402

ARM_RANGES = {}
for side, sgn in (("left", 1), ("right", -1)):
    ARM_RANGES.update({f"{side}_shoulder_pitch_joint": (-1.6, 0.6), f"{side}_shoulder_roll_joint": (sorted((0.0, 0.8 * sgn))),
                       f"{side}_shoulder_yaw_joint": (-0.6, 0.6), f"{side}_elbow_joint": (-1.0, 1.2),
                       f"{side}_wrist_roll_joint": (-0.5, 0.5), f"{side}_wrist_pitch_joint": (-0.5, 0.5),
                       f"{side}_wrist_yaw_joint": (-0.5, 0.5)})

REACH_RANGES = {}
for side, sgn in (("left", 1), ("right", -1)):
    REACH_RANGES.update({f"{side}_shoulder_pitch_joint": (-1.4, 0.0),
                         f"{side}_shoulder_roll_joint": tuple(sorted((-0.4 * sgn, 0.4 * sgn))),
                         f"{side}_shoulder_yaw_joint": tuple(sorted((-0.4 * sgn, 1.0 * sgn))), f"{side}_elbow_joint": (-0.2, 1.6),
                         f"{side}_wrist_roll_joint": (-0.3, 0.3), f"{side}_wrist_pitch_joint": (-0.3, 0.3),
                         f"{side}_wrist_yaw_joint": (-0.3, 0.3)})

G1_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF,
        fix_base=False,
        merge_fixed_joints=False,  # torso_link などの名前を残すため結合しない
        make_instanceable=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80), joint_pos={n: DEFAULT.get(n, 0.0) for n in LEGS + ["waist_yaw_joint"] + ARMS}),
    soft_joint_pos_limit_factor=0.95,
    # ゲイン・armature・トルク上限は MuJoCo と同じで、腕は陰的な減衰のまま（明示的だと手首が発振する）
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
            stiffness={".*_hip_.*": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_.*": 2.0, ".*_knee_joint": 4.0}, armature=0.01,
            effort_limit={".*_hip_.*": 88.0, ".*_knee_joint": 139.0}),
        "feet": IdealPDActuatorCfg(joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                                   stiffness=40.0, damping=2.0, armature=0.01, effort_limit=50.0),
        "waist": IdealPDActuatorCfg(joint_names_expr=["waist_yaw_joint"], stiffness=200.0, damping=5.0, armature=0.01,
                                    effort_limit=88.0),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
            stiffness={".*_shoulder_.*": 60.0, ".*_elbow_joint": 40.0, ".*_wrist_.*": 20.0},
            damping={".*_shoulder_.*": 2.0, ".*_elbow_joint": 2.0, ".*_wrist_.*": 1.0}, armature=0.01),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1,
                                 physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0))
    robot: ArticulationCfg = G1_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))


@configclass
class CommandsCfg:
    base_height = squat.BaseHeightCommandCfg(resampling_time_range=(2.5, 5.0), debug_vis=False)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=POLICY_JOINTS, scale=ACTION_SCALES,
                                           use_default_offset=True, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # MuJoCo でも計算できる量だけ
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=POLICY_JOINTS + ARMS, preserve_order=True)}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=POLICY_JOINTS + ARMS, preserve_order=True)}, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.6, 1.2),
        "dynamic_friction_range": (0.5, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 64})
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "mass_distribution_params": (-1.0, 2.0),
        "operation": "add"})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"yaw": (-3.14, 3.14)},
        "velocity_range": {"x": (-0.15, 0.15), "y": (-0.1, 0.1), "z": (0.0, 0.0), "yaw": (-0.8, 0.8)}})
    reset_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={
        "position_range": (-0.12, 0.12), "velocity_range": (-1.0, 1.0)})
    # 7 割は MuJoCo で集めた切り替え時の状態から始める
    reset_handoff = EventTerm(func=squat.reset_from_handoff_states, mode="reset", params={
        "path": "/project/lab/handoff_states.npz", "prob": 0.7})
    arm_targets = EventTerm(func=squat.randomize_arm_targets, mode="interval", interval_range_s=(1.0, 2.5), params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARMS, preserve_order=True), "ranges": ARM_RANGES,
        "reach_ranges": REACH_RANGES})
    leg_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS), "stiffness_distribution_params": (0.8, 1.2),
        "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
    push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(8.0, 12.0),
                           params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}})


@configclass
class RewardsCfg:
    track_height = RewTerm(func=squat.track_height_exp, weight=2.0, params={"command_name": "base_height", "std": 0.25})
    track_height_fine = RewTerm(func=squat.track_height_exp, weight=4.0, params={"command_name": "base_height", "std": 0.05})
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)  # 強すぎると、しゃがまないことを覚える
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_slide = RewTerm(func=loco_mdp.feet_slide, weight=-2.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    lin_vel_xy = RewTerm(func=squat.base_lin_vel_xy_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    yaw_rate = RewTerm(func=squat.base_yaw_rate_l2, weight=-0.5)
    feet_off_ground = RewTerm(func=squat.feet_off_ground, weight=-5.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")})
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)})
    hip_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", "waist_yaw_joint"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link", ".*_knee_link"]),
        "threshold": 1.0})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(50)})
    too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.28})


@configclass
class CurriculumCfg:
    squat_depth = CurrTerm(func=squat.height_curriculum, params={
        "command_name": "base_height", "start_low": 0.72, "end_low": 0.55, "num_steps": 24 * 300})


@configclass
class G1SquatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 0.002 秒では学習が遅すぎる
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.004
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1SquatEnvCfg_PLAY(G1SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
