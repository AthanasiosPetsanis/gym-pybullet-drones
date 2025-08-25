import argparse, os, sys, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# --- import του περιβάλλοντός σου ---
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

SUCCESS_THRESH = 0.30  # m, επιτυχία αν rel_goal norm < 0.30

# ---------- Wrappers (ίδιοι με το training) ----------

class AddDronePosWrapper(gym.ObservationWrapper):
    """
    Ενώνει: image + [rel_goal(3), drone_pos(3)]
    Δουλεύει είτε το env δίνει Dict(image, goal) είτε flat Box [image_flat + goal(3)].
    Πάντα επιστρέφει Dict: {"image": (3,256,256) float32 [0,1], "vec": (6,)}
    """
    def __init__(self, env: gym.Env, img_h: int = 256, img_w: int = 256):
        super().__init__(env)
        self.img_h, self.img_w, self.img_c = img_h, img_w, 3
        self.img_size = self.img_c * self.img_h * self.img_w
        self.is_dict_input = isinstance(env.observation_space, spaces.Dict)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0.0, 1.0, shape=(self.img_c, self.img_h, self.img_w), dtype=np.float32),
            "vec":   spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        })

    def _to_chw01(self, img):
        img = np.asarray(img)
        if img.ndim == 1:
            img = img.reshape(self.img_c, self.img_h, self.img_w)
        elif img.ndim == 3 and img.shape[-1] in (1,3):     # HWC
            img = np.transpose(img, (2,0,1))
        # else: assume already CHW
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img

    def observation(self, obs):
        base = self.env.unwrapped
        drone_pos = base._getDroneStateVector(0)[0:3].astype(np.float32)

        if self.is_dict_input:
            rel_goal = np.array(obs["goal"], dtype=np.float32)
            image = self._to_chw01(obs["image"])
        else:
            flat = np.asarray(obs, dtype=np.float32).flatten()
            img_flat = flat[:self.img_size]
            rel_goal = flat[self.img_size:self.img_size+3]
            image = self._to_chw01(img_flat)

        vec = np.concatenate([rel_goal, drone_pos]).astype(np.float32)
        return {"image": image, "vec": vec}


class DeltaToPIDRPM(gym.ActionWrapper):
    """
    Policy -> (dx,dy,dz) ∈ [-2,2]
    subgoal = cur_pos + delta
    DSLPID -> RPMs -> normalized [0..1] για VisionAviary.step()
    """
    def __init__(self, env: gym.Env, delta_limit: float = 2.0):
        super().__init__(env)
        self.delta_limit = float(delta_limit)
        self.action_space = spaces.Box(-self.delta_limit, self.delta_limit, shape=(3,), dtype=np.float32)

    def action(self, action):
        base = self.env.unwrapped
        delta = np.clip(action, -self.delta_limit, self.delta_limit).astype(np.float32)
        cur_pos = base._getDroneStateVector(0)[0:3]
        subgoal = (cur_pos + delta).astype(np.float32)

        try:
            rpms, _, _ = base.ctrl.computeControlFromState(
                control_timestep=base.control_timestep,
                state=base._getDroneStateVector(0),
                target_pos=subgoal,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )
        except Exception:
            rpms, _, _ = base.ctrl.computeControl(
                control_timestep=base.control_timestep,
                cur_state=base._getDroneStateVector(0),
                target_pos=subgoal,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )
        rpms = np.asarray(rpms, dtype=np.float32).flatten()
        return np.clip(rpms / 20000.0, 0.0, 1.0)  # VisionAviary περιμένει [0..1]


def make_env(gui=False, obstacles=True, seed=123):
    def _thunk():
        env = VisionAviary(gui=gui, record=False, obstacles=obstacles)
        env = Monitor(env)
        env = DeltaToPIDRPM(env, delta_limit=2.0)
        env = AddDronePosWrapper(env, img_h=256, img_w=256)
        env.reset(seed=seed)
        return env
    return _thunk


# ----------------- Evaluation loop -----------------

def evaluate(model_path: str, episodes: int, gui: bool, obstacles: bool, deterministic: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model on {device}: {model_path}")
    model = PPO.load(model_path, device=device)

    # Χρησιμοποιούμε single env (όχι Vec) για απλό loop
    env = make_env(gui=gui, obstacles=obstacles, seed=777)()

    ep_rewards, ep_lengths, successes, crashes = [], [], 0, 0

    for ep in range(1, episodes+1):
        obs, info = env.reset()
        done = False
        total_r, steps = 0.0, 0
        min_goal_dist = float("inf")
        while not done:
            # model.predict δέχεται Dict obs κανονικά
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_r += float(reward)
            steps += 1

            # Απόσταση προς τον κύριο στόχο από το obs["vec"][:3]
            if isinstance(obs, dict):
                goal_dist = float(np.linalg.norm(np.array(obs["vec"][:3], dtype=np.float32)))
                min_goal_dist = min(min_goal_dist, goal_dist)

            # αν το env περνά flag σύγκρουσης, αποθήκευσέ το
            if isinstance(info, dict) and ("collision" in info or "crashed" in info):
                if bool(info.get("collision", False) or info.get("crashed", False)):
                    crashes += 1

        success = (min_goal_dist < SUCCESS_THRESH)
        successes += int(success)
        ep_rewards.append(total_r)
        ep_lengths.append(steps)

        print(f"[EP {ep:02d}] R={total_r:8.3f} | steps={steps:3d} | min_goal_dist={min_goal_dist:5.3f} | "
              f"{'SUCCESS' if success else 'FAIL'}")

    # Σύνοψη
    print("\n==== SUMMARY ====")
    print(f"Episodes: {episodes}")
    print(f"Successes: {successes} / {episodes}  ({100.0*successes/episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(ep_rewards):.2f}   Std: {np.std(ep_rewards):.2f}")
    print(f"Mean Length: {np.mean(ep_lengths):.1f} steps")
    if crashes:
        print(f"Crashes (from info): ~{crashes} (best effort)")

    env.close()


def find_latest_zip(pattern_prefix="ppo_"):
    zips = [f for f in os.listdir(".") if f.startswith(pattern_prefix) and f.endswith(".zip")]
    if not zips:
        return None
    zips.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return zips[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to SB3 .zip (if omitted, pick latest ./ppo_*.zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--gui", action="store_true", help="Render with PyBullet GUI")
    parser.add_argument("--no-obstacles", action="store_true", help="Disable obstacles")
    parser.add_argument("--det", action="store_true", help="Deterministic actions")
    args = parser.parse_args()

    model_path = args.model or find_latest_zip()
    if not model_path or not os.path.isfile(model_path):
        print("[ERROR] Model .zip not found. Use --model path_to_zip")
        sys.exit(1)

    evaluate(model_path=model_path,
             episodes=args.episodes,
             gui=args.gui,
             obstacles=not args.no_obstacles,
             deterministic=args.det)