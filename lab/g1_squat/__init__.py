"""腕付き G1 が指令した高さまでしゃがむ・立つ Isaac Lab のタスク（G1-Squat-v0）"""
import gymnasium as gym

gym.register(
    id="G1-Squat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_squat.env_cfg:G1SquatEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_squat.agents:G1SquatPPORunnerCfg",
    },
)
gym.register(
    id="G1-Squat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_squat.env_cfg:G1SquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "g1_squat.agents:G1SquatPPORunnerCfg",
    },
)
