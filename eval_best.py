from stable_baselines3 import PPO
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

RUN_DIR = r".\results\save-08.27.2025_19.21.50"  # <-- άλλαξε σε δικό σου

env = VisionAviary(obs=ObservationType('rgb'), act=ActionType('pid'), gui=True, ctrl_freq=24)
model = PPO.load(RUN_DIR + r"\best_model.zip", device="cpu")

obs, _ = env.reset()
done = truncated = False
ret = 0.0
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, truncated, info = env.step(action)
    ret += r
print("Episode return:", ret)