import os, sys, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import math
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
# === IMPORT ENV (έχε κάνει editable install: pip install -e . στο root του project) ===
from gym_pybullet_drones.envs.VisionAviary import VisionAviary


# ---------------------------
# Observation wrapper
#  - Παίρνει το dict που δίνει το VisionAviary: {"image": (H,W,3) ή (3,H,W), "goal": (3,)}
#  - Προσθέτει την θέση drone και δίνει: {"image": (3,256,256) ή (256,256,3), "vec": (6,)}
# ---------------------------
class AddDronePosWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, img_h: int = 256, img_w: int = 256):
        super().__init__(env)
        self.img_h, self.img_w, self.img_c = img_h, img_w, 3
        self.img_size = self.img_c * self.img_h * self.img_w

        self.is_dict_input = isinstance(env.observation_space, spaces.Dict)

        # Ο unified χώρος εξόδου μας (CHW, float32, [0,1])
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(self.img_c, self.img_h, self.img_w), dtype=np.float32),
            "vec":   spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

    def _to_chw01(self, img):
        """Δέχεται είτε CHW είτε HWC, είτε uint8 είτε float; γυρίζει σε (3,H,W) float32 [0,1]."""
        img = np.array(img)
        if img.ndim == 3 and img.shape[0] in (1, 3):   # CHW
            chw = img
        elif img.ndim == 3 and img.shape[-1] in (1, 3):  # HWC
            chw = np.transpose(img, (2, 0, 1))
        elif img.ndim == 1:  # flat -> CHW
            chw = img.reshape(self.img_c, self.img_h, self.img_w)
        else:
            raise ValueError(f"Unsupported image shape {img.shape}")

        chw = chw.astype(np.float32)
        # αν είναι [0..255] κάνε scale στο [0..1]
        if chw.max() > 1.0:
            chw /= 255.0
        return chw

    def observation(self, obs):
        base = self.env.unwrapped
        drone_pos = base._getDroneStateVector(0)[0:3].astype(np.float32)

        if self.is_dict_input:
            # VisionAviary έδωσε Dict
            img_raw = obs["image"]
            rel_goal = np.array(obs["goal"], dtype=np.float32)
            image = self._to_chw01(img_raw)
        else:
            # VisionAviary έδωσε επίπεδο Box: [ image_flat + goal(3) ]
            flat = np.asarray(obs, dtype=np.float32).flatten()
            assert flat.size >= self.img_size + 3, \
                f"Flat obs length {flat.size} < expected {self.img_size+3}"
            img_flat = flat[:self.img_size]
            rel_goal = flat[self.img_size:self.img_size+3]
            image = self._to_chw01(img_flat)

        vec = np.concatenate([rel_goal, drone_pos]).astype(np.float32)
        return {"image": image, "vec": vec}


# ---------------------------
# Action wrapper
#  - Policy δίνει (dx,dy,dz) ∈ [-2,2]
#  - Υπολογίζουμε subgoal = cur_pos + delta
#  - Καλούμε DSLPID ΜΙΑ φορά (για αυτό το physics step)
#  - Επιστρέφουμε normalized RPMs [0..1] που περιμένει το VisionAviary
# ---------------------------
class DeltaToPIDRPM(gym.ActionWrapper):
    def __init__(self, env, delta_limit=2.0):
        super().__init__(env)
        self.delta_limit = float(delta_limit)
        self.action_space = spaces.Box(-self.delta_limit, self.delta_limit, shape=(3,), dtype=np.float32)
        self._SCALE_RPM = 20000.0   # <<< ΣΗΜΑΝΤΙΚΟ

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
                target_rpy_rates=np.zeros(3)
            )
        except Exception:
            rpms, _, _ = base.ctrl.computeControl(
                control_timestep=base.control_timestep,
                cur_state=base._getDroneStateVector(0),
                target_pos=subgoal,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3)
            )

        rpms = np.asarray(rpms, dtype=np.float32).flatten()
        # ---- σωστό normalize για το δικό σου VisionAviary ----
        norm = np.clip(rpms / self._SCALE_RPM, 0.0, 1.0).astype(np.float32)
        return norm

class MacroPIDWaypointWrapper(gym.Wrapper):
    """
    Policy action: (dx,dy,dz) ∈ [-delta_limit, +delta_limit]
    Σε ΚΑΘΕ RL-step: εκτελεί μέχρι max_pid_steps κλήσεις του DSLPID προς τον υποστόχο,
    με ύψος-αντιστάθμιση (altitude compensation) ~ 1/cos(tilt), και normalize σε [0..1].
    """
    def __init__(self, env, delta_limit=1.0, subgoal_eps=0.20, max_pid_steps=8,
                 assist_tilt=True, tilt_gain=0.4, tilt_max_deg=15.0,
                 boost_altitude=True, boost_max=1.10, scale_rpm=21700.0):
        super().__init__(env)
        self.delta_limit = float(delta_limit)
        self.subgoal_eps = float(subgoal_eps)
        self.max_pid_steps = int(max_pid_steps)
        self.assist_tilt = assist_tilt
        self.tilt_gain = float(tilt_gain)
        self.tilt_max = math.radians(tilt_max_deg)
        self.boost_altitude = bool(boost_altitude)
        self.boost_max = float(boost_max)
        self.scale_rpm = float(scale_rpm)
        self.action_space = spaces.Box(low=-self.delta_limit, high=self.delta_limit,
                                       shape=(3,), dtype=np.float32)

    def _pid_to_norm(self, target_pos):
        base = self.env.unwrapped
        
        cur_state = base._getDroneStateVector(0)
        cur_pos = cur_state[0:3]

        # --- μικρή επιθυμητή κλίση προς το σφάλμα XY (roll, pitch) ---
        if self.assist_tilt:
            ex, ey = float(target_pos[0] - cur_pos[0]), float(target_pos[1] - cur_pos[1])
            # mapping αξόνων: roll ~ -ey, pitch ~ +ex (αν δεις ανάποδη κίνηση, αντέστρεψέ τα)
            roll_cmd  = np.clip(-self.tilt_gain * ey, -self.tilt_max, self.tilt_max)
            pitch_cmd = np.clip( +self.tilt_gain * ex, -self.tilt_max, self.tilt_max)
            target_rpy = np.array([roll_cmd, pitch_cmd, 0.0], dtype=np.float32)
        else:
            target_rpy = np.zeros(3, dtype=np.float32)
        # PID -> RPMs
        try:
            rpms, _, _ = base.ctrl.computeControlFromState(
                control_timestep=base.control_timestep,
                state=base._getDroneStateVector(0),
                target_pos=target_pos,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )
        except Exception:
            rpms, _, _ = base.ctrl.computeControl(
                control_timestep=base.control_timestep,
                cur_state=base._getDroneStateVector(0),
                target_pos=target_pos,
                target_rpy=np.zeros(3),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )
        rpms = np.asarray(rpms, dtype=np.float32).flatten()

        # Altitude compensation: 1/cos(tilt), με σωστούς δείκτες roll=state[6], pitch=state[7]
        if self.boost_altitude:
            cur = base._getDroneStateVector(0)[0:3]
            xy_err = float(np.linalg.norm(target_pos[:2] - cur[:2]))
            if xy_err > 0.10:  # εφαρμόζουμε boost μόνο αν όντως πάμε για XY
                state = base._getDroneStateVector(0)
                if state.size > 9:
                    roll, pitch = float(state[6]), float(state[7])  # rad  ✅ σωστοί δείκτες
                    tilt = math.sqrt(roll*roll + pitch*pitch)
                    boost = 1.0 / max(0.01, math.cos(min(tilt, math.radians(50))))
                    rpms = rpms * float(np.clip(boost, 1.0, self.boost_max))
        cur_pos = base._getDroneStateVector(0)[0:3]
        ex = float(target_pos[0] - cur_pos[0])   # σφάλμα στον άξονα x
        ey = float(target_pos[1] - cur_pos[1])   # σφάλμα στον άξονα y
        kx = 0.035   # πόσο pitch για 1m σφάλμα στο x
        ky = 0.035   # πόσο roll  για 1m σφάλμα στο y
        pitch_w = np.array([-1.0, +1.0, +1.0, -1.0], dtype=np.float32)  # προς +x
        roll_w  = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.float32)  # προς +y

        # offset σε RPMs (όχι normalized) — περιορίζουμε για ασφάλεια
        rpm_offset = (kx * ex) * pitch_w + (ky * ey) * roll_w
        #scale το offset σε RPM μονάδες (π.χ. 300 RPM max προσθήκη)
        rpm_offset = np.clip(rpm_offset * 800.0, -800.0, +800.0)

        rpms = rpms + rpm_offset
        rpms = np.clip(rpms, 0.0, self.scale_rpm * 1.05)  # λίγη «ανάσα» πάνω απ’ τη scale πριν το normalize
        # Normalize για VisionAviary: [0..1] με κλίμακα ~20000
        act = np.clip(rpms / self.scale_rpm, 0.0, 1.0).astype(np.float32)
        return act

    def step(self, delta):
        base = self.env.unwrapped
        delta = np.clip(delta, -1.0, 1.0).astype(np.float32)
        cur = base._getDroneStateVector(0)[0:3]
        subgoal = (cur + delta).astype(np.float32)

        total_r, done_term, done_trunc = 0.0, False, False
        last_obs, last_info = None, {}

        # μικρός «μακρο-βρόχος» PID: δίνουμε χρόνο να πιάσει τον υποστόχο
        for _ in range(self.max_pid_steps):
            act = self._pid_to_norm(subgoal)
            obs, r, term, trunc, info = self.env.step(act)
            total_r += float(r); done_term |= bool(term); done_trunc |= bool(trunc)
            last_obs, last_info = obs, info
            new_pos = base._getDroneStateVector(0)[0:3]
            if float(np.linalg.norm(new_pos - subgoal)) <= self.subgoal_eps or done_term or done_trunc:
                break

        return last_obs, total_r, done_term, done_trunc, last_info
    
# ---------------------------
# Features Extractor (CNN για εικόνα + MLP για vec)
#  - Δέχεται είτε HWC είτε CHW. Αν χρειαστεί, μεταθέτει σε CHW.
#  - Κανονικοποιεί εικόνα σε [0,1] αν είναι uint8.
# ---------------------------
class ProgressRewardWrapper(gym.Wrapper):
    def __init__(self, env, k_progress=1.0, step_penalty=0.005):
        super().__init__(env)
        self.k = float(k_progress)
        self.step_penalty = float(step_penalty)
        self._last_goal_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict):
            self._last_goal_dist = float(np.linalg.norm(np.array(obs["vec"][:3], dtype=np.float32)))
        else:
            self._last_goal_dist = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, dict):
            curr = float(np.linalg.norm(np.array(obs["vec"][:3], dtype=np.float32)))
            if isinstance(info, dict):
                info = dict(info)  # copy
            else:
                info = {}
            info["dbg_goal_dist"] = curr
            info["dbg_success"] = float(curr < 0.30)
            prev = self._last_goal_dist
            if prev is not None:
                reward += self.k * (prev - curr)  # πρόοδος
            self._last_goal_dist = curr

        reward -= self.step_penalty
        return obs, reward, terminated, truncated, info
    
class TBSimpleMetrics(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "dbg_goal_dist" in info:
                self.logger.record("custom/goal_dist", info["dbg_goal_dist"])
            if "dbg_success" in info:
                self.logger.record("custom/success", info["dbg_success"])
        return True
    
class CNNPlusVecExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        img_space = observation_space["image"]
        self.is_hwc = (len(img_space.shape) == 3 and img_space.shape[-1] in (1, 3))
        c, h, w = (img_space.shape[-1], img_space.shape[0], img_space.shape[1]) if self.is_hwc \
                  else (img_space.shape[0], img_space.shape[1], img_space.shape[2])

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros((1, c, h, w))
            cnn_out = self.cnn(sample).shape[1]

        self.vec_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_out + 64, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, obs):
        img = obs["image"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if self.is_hwc:
            # (B,H,W,C) -> (B,C,H,W)
            img = img.permute(0, 3, 1, 2).contiguous()
        vec = obs["vec"].float()
        f_img = self.cnn(img)
        f_vec = self.vec_mlp(vec)
        return self.head(torch.cat([f_img, f_vec], dim=1))


# ---------------------------
# Env factory (απλό, 1 env)
# ---------------------------
def make_env(gui=False, obstacles=True, seed=42):
    def _thunk():
        env = VisionAviary(gui=gui, record=False, obstacles=obstacles)
        env = Monitor(env)
        env = MacroPIDWaypointWrapper(env, delta_limit=2.0, subgoal_eps=0.20, max_pid_steps=12,
                                      boost_altitude=True, boost_max=1.35, scale_rpm=20000.0,)     # (dx,dy,dz) -> DSLPID -> RPMs
        env = AddDronePosWrapper(env)                 # προσθέτει drone_pos στο goal vector
        env = ProgressRewardWrapper(env, k_progress=1.0, step_penalty=0.005)
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    log_dir = "./ppo_vision_tensorboard"
    os.makedirs(log_dir, exist_ok=True)

    # CUDA αν υπάρχει (αλλιώς CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] SB3 device -> {device_str}")

    # Ένα απλό VecEnv
    env = DummyVecEnv([make_env(gui=False, obstacles=True, seed=42)])

    policy_kwargs = dict(
        features_extractor_class=CNNPlusVecExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64])  # απλό MLP head
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,          # γρήγορα πρώτα updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device=device_str,
        verbose=1,
        seed=42
    )

    model.learn(total_timesteps=300_000, log_interval=1, progress_bar=True, tb_log_name="cuda_run", callback=TBSimpleMetrics())
    model.save(f"./ppo_vision_delta2pid_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    print("Training finished.")


if __name__ == "__main__":
    main()