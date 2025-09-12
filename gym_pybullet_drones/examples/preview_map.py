# preview_map.py
import time, numpy as np
import pybullet as p
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

def hover_action(env):
    # PID expects absolute position setpoints per drone: shape (NUM_DRONES, 3)
    pos = env._getDroneStateVector(0)[0:3]
    return pos[None, :].astype(np.float32)  # (1,3)

def show(difficulty=0, seconds=8):
    env = VisionAviary(gui=True, obs=ObservationType('rgb'),
                       act=ActionType('pid'), ctrl_freq=24, difficulty=difficulty)
    obs, info = env.reset()
    p.addUserDebugText(f"DIFFICULTY {difficulty}", [0,0,1.8], textColorRGB=[1,1,0],
                       lifeTime=seconds, physicsClientId=env.CLIENT)

    start = time.time()
    steps = int(seconds / env.CTRL_TIMESTEP)
    for _ in range(steps):
        obs, r, done, trunc, info = env.step(hover_action(env))
        env.render()
        sync(env.step_counter, start, env.CTRL_TIMESTEP)
    env.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    show(args.difficulty, args.seconds)
