"""
Minimal PPO + VisionAviary (image obs, PID actions) trainer/evaluator.
Saves best_model.zip under results/save-<timestamp>/
"""

import os, time
from datetime import datetime
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
tmp = VisionAviary(obs=ObservationType('rgb'), act=ActionType('pid'), ctrl_freq=24, gui=False)
print("OBS SPACE:", tmp.observation_space, tmp.observation_space.dtype)
obs, _ = tmp.reset()
print("OBS SHAPE/DTYPE:", getattr(obs, "shape", None), getattr(obs, "dtype", None))
tmp.close()
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType('rgb')     # image obs
DEFAULT_ACT = ActionType('pid')          # 3D setpoint -> PID -> RPMs
DEFAULT_MA  = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=False, colab=False, record_video=DEFAULT_RECORD_VIDEO, local=True):
    # ---- paths ----
    run_dir = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(run_dir, exist_ok=True)

    # ---- envs ----
    train_env = make_vec_env(
        HoverAviary,
        env_kwargs=dict(
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            ctrl_freq=24,
            gui=True                 # if you want the PyBullet window
            # render_mode="human"       # or "rgb_array" if you only need frames
        ),
        n_envs=1
    )
    eval_env = VisionAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=24, gui=False)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # ---- callbacks (define BEFORE learn) ----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir + '/',
        log_path=run_dir + '/',
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )

    # ---- model ----
    model = PPO(
        "MlpPolicy",            # <-- change from CnnPolicy
        train_env,
        device="cpu",           # MLP runs better on CPU
        verbose=1
    )

    # ---- train ----
    try:
        model.learn(total_timesteps=int(3e5), callback=eval_callback)
    except KeyboardInterrupt:
        model.save(run_dir + '/interrupt_model')
        print('Saved:', run_dir + '/interrupt_model.zip')

    # ---- final save ----
    model.save(run_dir + '/final_model.zip')
    print('Saved run to:', run_dir)

    # ---- quick numeric summary ----
    if os.path.isfile(run_dir + '/evaluations.npz'):
        with np.load(run_dir + '/evaluations.npz') as data:
            print('Last eval @', data['timesteps'][-1], 'steps, mean_reward =', data['results'].mean(axis=1)[-1])

    # ---- visual eval (loads best_model.zip if present) ----
    model_path = run_dir + '/best_model.zip' if os.path.isfile(run_dir + '/best_model.zip') else run_dir + '/final_model.zip'
    model = PPO.load(model_path, device="cpu")

    test_env = VisionAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=24, record=record_video)
    obs, info = test_env.reset(seed=42)
    done = truncated = False
    start = time.time()
    ep_ret = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = test_env.step(action)
        ep_ret += r
        test_env.render()
        sync(test_env.step_counter, start, test_env.CTRL_TIMESTEP)
    print('Episode return:', ep_ret)
    test_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--colab', default=False, type=bool)
    ARGS = parser.parse_args()
    run(**vars(ARGS))