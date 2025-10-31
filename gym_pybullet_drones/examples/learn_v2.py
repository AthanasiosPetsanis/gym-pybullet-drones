import os
from datetime import datetime
import argparse
import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "ddpg"])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--gui", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--difficulty", type=int, default=2)  # ← level 2: single cylinder
    p.add_argument("--n-envs", type=int, default=8)      # parallel envs
    p.add_argument("--repeat-k", type=int, default=3)    # action repeat (PID responsiveness vs speed)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--ckpt-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# --------- small wrapper: ActionRepeat ----------
class ActionRepeat(gym.Wrapper):
    def __init__(self, env, k: int = 3):
        super().__init__(env)
        self.k = int(k)

    def step(self, action):
        total_r = 0.0
        term = trunc = False
        info = {}
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(action)
            total_r += r
            if term or trunc:
                break
        return obs, total_r, term, trunc, info


def make_env_fn(difficulty, ctrl_hz=24, gui=False, repeat_k=3):
    def _f():
        env = VisionAviary(
            gui=gui,
            obs=ObservationType('kin'),     # KIN + PID (fast baseline)
            act=ActionType('pid'),
            ctrl_freq=ctrl_hz,
            difficulty=difficulty,
            random_start=True,
            start_center_xy=(0.0, 0.0),
            start_radius=1.2,
            start_z_range=(0.75, 0.95),
            keep_goal_z_equal_spawn=True,
        )
        env = ActionRepeat(env, k=repeat_k)
        return env
    return _f


def build_vec_env(n_envs, difficulty, gui, repeat_k, seed):
    env_fns = [make_env_fn(difficulty, gui=False, repeat_k=repeat_k) for _ in range(n_envs)]
    venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv.seed(seed)
    return VecMonitor(venv)


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    run_dir = os.path.join("results", "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] Training {args.algo.upper()} | diff={args.difficulty} | n_envs={args.n_envs} | k={args.repeat_k}")
    print(f"[INFO] Logging to: {run_dir}")

    # -------- Vec envs --------
    train_env = build_vec_env(args.n_envs, args.difficulty, args.gui, args.repeat_k, args.seed)
    eval_env = build_vec_env(1, args.difficulty, False, args.repeat_k, args.seed + 1)

    # -------- Callbacks (quiet terminal, TB always on) --------
    callbacks = [
        CheckpointCallback(save_freq=args.ckpt_freq, save_path=run_dir, name_prefix="ckpt"),
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=0,
        ),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Algorithms --------
    if args.algo.lower() == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            device="cpu",
            tensorboard_log=run_dir,  # TB always on
            learning_rate=3e-4,
            n_steps=512,              # with 8 envs → 4096 per rollout
            batch_size=2048,          # big batches for speed/stability
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=args.seed,
        )
    elif args.algo.lower() == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            device=device,
            tensorboard_log=run_dir,
            learning_rate=2e-4,
            buffer_size=200_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
            ent_coef="auto",
            tau=0.005,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=args.seed,
        )
    elif args.algo.lower() == "ddpg":
        model = DDPG(
            policy="MlpPolicy",
            env=train_env,
            device=device,
            tensorboard_log=run_dir,
            learning_rate=1e-3,
            buffer_size=200_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown algo {args.algo}")

    # -------- Learn --------
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(run_dir, "final_model.zip"))

    print("[OK] Done. Saved to:", run_dir)
    print("TensorBoard:", f"tensorboard --logdir {run_dir}")


if __name__ == "__main__":
    main()