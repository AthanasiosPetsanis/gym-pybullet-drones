"""
Minimal PPO + VisionAviary (image obs, PID actions) trainer/evaluator.
Saves best_model.zip under results/save-<timestamp>/
"""

import os, time
from datetime import datetime
import argparse
import numpy as np
import os, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType('kin')     # image obs
DEFAULT_ACT = ActionType('pid')          # 3D setpoint -> PID -> RPMs
DEFAULT_MA  = False
DIFFICULTY = 2  # 0: easy, 1: medium, 2: hard

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = int(k)

    def step(self, action):
        total_r = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self.k):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_r += r
            if terminated or truncated:
                break
        return obs, total_r, terminated, truncated, info
def plot_evals(run_dir, show=True, save=True, title_suffix=""):
    import os, numpy as np, matplotlib.pyplot as plt
    evf = os.path.join(run_dir, "evaluations.npz")
    if not os.path.isfile(evf):
        print("[WARN] No evaluations.npz found; nothing to plot."); return
    E = np.load(evf)
    t = E["timesteps"]
    R = E["results"]      # (points, n_eval_eps)
    L = E["ep_lengths"]   # (points, n_eval_eps)
    r_mean, r_std = R.mean(axis=1), R.std(axis=1)
    l_mean = L.mean(axis=1)

    plt.figure()
    plt.plot(t, r_mean); plt.fill_between(t, r_mean-r_std, r_mean+r_std, alpha=0.2)
    plt.xlabel("timesteps"); plt.ylabel("eval mean reward (±std)")
    plt.title(f"PPO (KIN) — reward vs timesteps{title_suffix}"); plt.grid(True)
    if save: plt.savefig(os.path.join(run_dir, "eval_mean_reward.png"), dpi=150)

    plt.figure()
    plt.plot(t, l_mean)
    plt.xlabel("timesteps"); plt.ylabel("eval mean episode length")
    plt.title(f"PPO (KIN) — ep length vs timesteps{title_suffix}"); plt.grid(True)
    if save: plt.savefig(os.path.join(run_dir, "eval_mean_ep_len.png"), dpi=150)

    if show: plt.show()
    else: plt.close('all')

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI,
        plot=True, colab=False, record_video=DEFAULT_RECORD_VIDEO, local=True,
        save_plots=True, eval_episodes=10, demo_seconds=8.0):
    # ---- paths ----
    run_dir = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(run_dir, exist_ok=True)

    # --- TRAIN ENV ---
    train_env = make_vec_env(
        VisionAviary,
        env_kwargs=dict(
            obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=24, difficulty=DIFFICULTY,
            random_start=True, start_center_xy=(0.0, 0.0), start_radius=1.2,
            start_z_range=(0.75, 0.95), keep_goal_z_equal_spawn=True,
        ),
        n_envs=1,
        wrapper_class=ActionRepeat,          # <<< add this
        wrapper_kwargs=dict(k=4),            # <<< and this (try k=4; later tune 3–5)
    )
    train_env = VecFrameStack(train_env, n_stack=4, channels_order='first')

    # --- EVAL ENV ---
    eval_env = make_vec_env(
        VisionAviary,
        env_kwargs=dict(
            obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=24, difficulty=DIFFICULTY,
            random_start=True, start_center_xy=(0.0, 0.0), start_radius=1.2,
            start_z_range=(0.75, 0.95), keep_goal_z_equal_spawn=True,
        ),
        n_envs=1,
        wrapper_class=ActionRepeat,          # <<< same wrapper for eval
        wrapper_kwargs=dict(k=4),
    )
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='first')


    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # ---- callbacks (define BEFORE learn) ----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir + '/',
        log_path=run_dir + '/',
        eval_freq=10000,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        callback_on_new_best=None
    )

    # ---- model ----
    
    model = SAC(
        "MlpPolicy", train_env,
        device="cpu",
        learning_rate=2e-4,       # modest LR
        buffer_size=200_000,      # big enough, not huge
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,             # 1 gradient step per env step
        gradient_steps=1,         # keep it light/fast
        learning_starts=1_000,    # start learning quickly
        ent_coef="auto",          # automatic entropy tuning
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256])  # small, fast MLP
    )

    # ---- train ----
    try:
        model.learn(total_timesteps=int(1e6), callback=eval_callback)
    except KeyboardInterrupt:
        model.save(run_dir + '/interrupt_model')
        print('Saved:', run_dir + '/interrupt_model.zip')

    # ---- final save ----
    model.save(run_dir + '/final_model.zip')
    print('Saved run to:', run_dir)

    # ---- plots ----
    plot_evals(run_dir, show=plot, save=save_plots, title_suffix=f" (diff={DIFFICULTY})")

    if gui and demo_seconds > 0:
        obs, _ = test_env.reset()
        import time
        start_demo = time.time()
        while time.time() - start_demo < demo_seconds:
            pos = test_env._getDroneStateVector(0)[0:3]
            action = pos[None, :].astype(np.float32)  # PID setpoint (1,3)
            obs, _, _, _, _ = test_env.step(action)
            test_env.render()
            sync(test_env.step_counter, start_demo, test_env.CTRL_TIMESTEP)
    # ---- quick numeric summary ----
    if os.path.isfile(run_dir + '/evaluations.npz'):
        with np.load(run_dir + '/evaluations.npz') as data:
            print('Last eval @', data['timesteps'][-1], 'steps, mean_reward =', data['results'].mean(axis=1)[-1])

    # ---- visual eval (loads best_model.zip if present) ----
    model_path = run_dir + '/best_model.zip' if os.path.isfile(run_dir + '/best_model.zip') else run_dir + '/final_model.zip'
    AlgoClass = type(model)                 # reuse the class you just trained (PPO/SAC/DDPG/...)
    model = AlgoClass.load(model_path, device="cpu")

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
    parser.add_argument('--plot', type=bool, default=True)          # show plots in a window
    parser.add_argument('--save_plots', type=bool, default=True)    # also save PNGs
    parser.add_argument('--eval_episodes', type=int, default=10)    # per evaluation
    parser.add_argument('--demo_seconds', type=float, default=8.0)
    ARGS = parser.parse_args()
    run(**vars(ARGS))