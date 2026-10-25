"""学習・書き出し・MuJoCo 側で共有する定義（関節の並び・既定の姿勢・観測の係数）"""

LEGS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
ARMS = [f"{s}_{j}_joint" for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
POLICY_JOINTS = LEGS + ["waist_yaw_joint"]
OBS_JOINTS = POLICY_JOINTS + ARMS
# unitree_rl_gym の g1.yaml と同じ
DEFAULT = {"left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1, "left_knee_joint": 0.3, "right_knee_joint": 0.3,
           "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2}
ACTION_SCALE = 0.75
# ひねりは固定する（動かせると体ごと回る動きを覚える）
FIXED_JOINTS = ["left_hip_yaw_joint", "right_hip_yaw_joint", "waist_yaw_joint"]
ACTION_SCALES = {n: 0.0 if n in FIXED_JOINTS else ACTION_SCALE for n in POLICY_JOINTS}
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
STAND_HEIGHT = 0.78
HEIGHT_RANGE = (0.55, 0.80)
# 観測: 角速度(3)×0.25, 重力(3), 高さの指令(1), 関節角−既定(27), 関節速度(27)×0.05, 前回の行動(13) = 74
OBS_DIM = 3 + 3 + 1 + len(OBS_JOINTS) * 2 + len(POLICY_JOINTS)
