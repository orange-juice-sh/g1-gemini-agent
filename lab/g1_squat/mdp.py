from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass


class BaseHeightCommand(CommandTerm):
    """腰の高さの指令
    目標へゆっくり近づける"""

    cfg: "BaseHeightCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height = torch.full((self.num_envs, 1), cfg.stand_height, device=self.device)
        self.goal = self.height.clone()
        self.step_dt = env.step_dt
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height

    def _update_metrics(self):
        self.metrics["error_height"] = torch.abs(self.robot.data.root_pos_w[:, 2] - self.height[:, 0])

    def _resample_command(self, env_ids):
        n = len(env_ids)
        lo, hi = self.cfg.height_range
        h = torch.rand(n, device=self.device) * (hi - lo) + lo
        stand = torch.rand(n, device=self.device) < self.cfg.stand_prob
        h[stand] = self.cfg.stand_height
        self.goal[env_ids, 0] = h

    def _update_command(self):
        dh = torch.clamp(self.goal - self.height, -self.cfg.rate * self.step_dt, self.cfg.rate * self.step_dt)
        self.height += dh

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.height[env_ids] = self.cfg.stand_height
        return super().reset(env_ids)


@configclass
class BaseHeightCommandCfg(CommandTermCfg):
    class_type: type = BaseHeightCommand
    asset_name: str = "robot"
    height_range: tuple = (0.40, 0.80)
    stand_height: float = 0.78
    stand_prob: float = 0.3
    rate: float = 0.4


def track_height_exp(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(robot.data.root_pos_w[:, 2] - target) / std**2)


def height_curriculum(env, env_ids, command_name: str, start_low: float, end_low: float, num_steps: int):
    """しゃがむ深さの下限を start_low から end_low へ広げる"""
    term = env.command_manager.get_term(command_name)
    frac = min(1.0, env.common_step_counter / num_steps)
    low = start_low + (end_low - start_low) * frac
    term.cfg.height_range = (low, term.cfg.height_range[1])
    return low


def feet_off_ground(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (forces < threshold).float().sum(dim=1)


def base_lin_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)


def base_yaw_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_b[:, 2])


def randomize_arm_targets(env, env_ids, asset_cfg: SceneEntityCfg, ranges: dict, reach_ranges: dict | None = None):
    """腕の目標角を不規則に変える（半分は両腕を前に伸ばす姿勢）"""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = asset_cfg.joint_ids
    lo = torch.tensor([ranges[n][0] for n in asset_cfg.joint_names], device=env.device)
    hi = torch.tensor([ranges[n][1] for n in asset_cfg.joint_names], device=env.device)
    targets = torch.rand(len(env_ids), len(ids), device=env.device) * (hi - lo) + lo
    if reach_ranges is not None:
        rlo = torch.tensor([reach_ranges[n][0] for n in asset_cfg.joint_names], device=env.device)
        rhi = torch.tensor([reach_ranges[n][1] for n in asset_cfg.joint_names], device=env.device)
        reach = torch.rand(len(env_ids), len(ids), device=env.device) * (rhi - rlo) + rlo
        use = torch.rand(len(env_ids), 1, device=env.device) < 0.5
        targets = torch.where(use, reach, targets)
    robot.set_joint_position_target(targets, joint_ids=ids, env_ids=env_ids)


def reset_from_handoff_states(env, env_ids, path: str, prob: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """prob の割合の試行を、MuJoCo で集めた切り替え時の状態から始める（向きは乱数）"""
    import numpy as np
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

    robot: Articulation = env.scene[asset_cfg.name]
    cache = getattr(reset_from_handoff_states, "_cache", None)
    if cache is None:
        z = np.load(path)
        t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=env.device)  # noqa: E731
        joint_ids, _ = robot.find_joints([str(n) for n in z["joint_names"]], preserve_order=True)
        cache = dict(height=t("height"), rp=t("roll_pitch"), lin=t("lin_vel_b"), ang=t("ang_vel_b"),
                     q=t("joint_pos"), dq=t("joint_vel"), joint_ids=joint_ids)
        reset_from_handoff_states._cache = cache
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(ids) == 0:
        return
    k = torch.randint(0, len(cache["height"]), (len(ids),), device=env.device)
    yaw = (torch.rand(len(ids), device=env.device) * 2 - 1) * math.pi
    quat = quat_from_euler_xyz(cache["rp"][k, 0], cache["rp"][k, 1], yaw)
    pos = env.scene.env_origins[ids].clone()
    pos[:, 2] += cache["height"][k]
    lin_w = quat_apply(quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), cache["lin"][k])
    ang_w = quat_apply(quat, cache["ang"][k])  # 体の座標系
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=1), env_ids=ids)
    q = robot.data.default_joint_pos[ids].clone()
    dq = torch.zeros_like(q)
    q[:, cache["joint_ids"]] = cache["q"][k]
    dq[:, cache["joint_ids"]] = cache["dq"][k]
    robot.write_joint_state_to_sim(q, dq, env_ids=ids)
