import os, glob, argparse, numpy as np
import matplotlib.pyplot as plt
from collections import deque

# SB3
try:
    from stable_baselines3 import PPO, SAC, DDPG
except Exception:
    PPO = SAC = DDPG = None

# Enums (both enum and factory-style supported)
ObservationType = ActionType = None
try:
    from gym_pybullet_drones.utils.enums import ObservationType as _OT, ActionType as _AT
    ObservationType, ActionType = _OT, _AT
except Exception:
    try:
        from gym_pybullet_drones.envs.BaseAviary import ObservationType as _OT, ActionType as _AT
        ObservationType, ActionType = _OT, _AT
    except Exception:
        pass

from gym_pybullet_drones.envs.VisionAviary import VisionAviary

def coerce_obs(v="kin"):
    if ObservationType is None: return v
    try: return ObservationType(v)  # factory
    except Exception:
        try: return ObservationType.KIN if v.lower()=="kin" else ObservationType.RGB
        except Exception: return v

def coerce_act(v="pid"):
    if ActionType is None: return v
    try: return ActionType(v)
    except Exception:
        try: return ActionType.PID
        except Exception: return v

def newest_run(root="results"):
    runs = sorted(glob.glob(os.path.join(root, "save-*")))
    if not runs: raise FileNotFoundError("No runs in results/save-*")
    return runs[-1]

def load_model(run_dir, prefer="best"):
    mp = os.path.join(run_dir, "best_model.zip" if prefer=="best" else "final_model.zip")
    if not os.path.isfile(mp):
        alt = os.path.join(run_dir, "final_model.zip")
        if os.path.isfile(alt): mp = alt
        else: raise FileNotFoundError(f"No model zip in {run_dir}")
    for cls in (PPO, SAC, DDPG):
        if cls is None: continue
        try: return cls.load(mp), cls.__name__, mp
        except Exception: pass
    raise RuntimeError(f"Could not load {mp} with PPO/SAC/DDPG")

def squeeze(o):
    arr = np.asarray(o)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr.squeeze(0)
    return arr

class FrameStacker:
    def __init__(self, k, single_shape):
        self.k = int(k); self.single_shape = tuple(single_shape); self.buf = deque(maxlen=self.k)
    def reset(self, first_obs):
        x = squeeze(first_obs)
        for _ in range(self.k): self.buf.append(x.copy())
        return self.get()
    def step(self, obs):
        x = squeeze(obs); self.buf.append(x.copy()); return self.get()
    def get(self):
        return np.stack(list(self.buf), axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--prefer", choices=["best","final"], default="best")
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    run_dir = newest_run()
    model, algo, mp = load_model(run_dir, args.prefer)
    print(f"Loaded {algo}: {mp}")

    env = VisionAviary(gui=args.gui, obs=coerce_obs("kin"), act=coerce_act("pid"),
                       ctrl_freq=24, difficulty=args.difficulty,
                       random_start=True, keep_goal_z_equal_spawn=True)
    obs, info = env.reset()

    # detect stacking
    exp = getattr(model, "observation_space", None)
    if exp is None: raise RuntimeError("Model has no observation_space")
    exp_shape = tuple(exp.shape)        # e.g., (48,) or (4,48)
    curr = squeeze(obs)
    stacker = None
    if len(exp_shape)>=2 and curr.shape != exp_shape and exp_shape[1:] == curr.shape:
        stacker = FrameStacker(exp_shape[0], curr.shape)
        curr = stacker.reset(curr)

    # rollout
    pos_log, sp_log = [], []
    goal = env.goal.copy()
    done = False; t=0

    if args.gui:
        try:
            import pybullet as p
            p.setRealTimeSimulation(0)
        except Exception: pass

    while not done and t < args.max_steps:
        act, _ = model.predict(curr, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)

        # primary: read from info
        if "pos" in info: pos_log.append(np.array(info["pos"], dtype=float))
        else:  # fallback: read directly
            pos_log.append(np.array(env._getDroneStateVector(0)[0:3], dtype=float))
        if "pid_target" in info: sp_log.append(np.array(info["pid_target"], dtype=float))
        else:
            # fallback to attribute if user added it but didn't export in info
            if hasattr(env, "_last_pid_target"):
                sp_log.append(np.array(env._last_pid_target, dtype=float))

        nxt = squeeze(obs)
        curr = stacker.step(nxt) if stacker else nxt
        t += 1
        if args.gui: env.render()

    env.close()

    pos = np.stack(pos_log, axis=0) if pos_log else np.zeros((0,3))
    sp  = np.stack(sp_log,  axis=0) if sp_log  else np.zeros((0,3))

    outdir = os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)

    # plot
    fig = plt.figure(figsize=(8,6.5))
    ax = fig.add_subplot(111, projection="3d")
    if len(pos):
        ax.plot(pos[:,0], pos[:,1], pos[:,2], label="Drone path (PID result)", linewidth=2)
        ax.scatter(pos[0,0], pos[0,1], pos[0,2], s=40, label="Start")
    if len(sp):
        ax.plot(sp[:,0], sp[:,1], sp[:,2], "--", alpha=0.85, label="Policy setpoints (to PID)")
        idx = np.linspace(0, len(sp)-1, num=min(40, len(sp))).astype(int)
        ax.scatter(sp[idx,0], sp[idx,1], sp[idx,2], s=14, alpha=0.7)
    ax.scatter([goal[0]],[goal[1]],[goal[2]], c="g", s=60, label="Goal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Level {args.difficulty}: Path vs Setpoints ({algo})")
    ax.legend(loc="best"); plt.tight_layout()

    png = os.path.join(outdir, f"path_vs_setpoints_lvl{args.difficulty}.png")
    plt.savefig(png, dpi=150)
    csv = os.path.join(outdir, f"path_vs_setpoints_lvl{args.difficulty}.csv")
    print("Saved plot:", os.path.abspath(png))

    # CSV
    T = max(len(pos), len(sp))
    def pad(a,n):
        if len(a)>=n: return a[:n]
        if a.ndim==2: return np.vstack([a, np.full((n-len(a), a.shape[1]), np.nan)])
        return np.concatenate([a, np.full((n-len(a),), np.nan)])
    pos_p, sp_p = pad(pos, T), pad(sp, T)
    tcol = np.arange(T).reshape(-1,1)
    arr = np.hstack([tcol,
                     pos_p if pos_p.size else np.full((T,3), np.nan),
                     sp_p  if sp_p.size  else np.full((T,3), np.nan)])
    np.savetxt(csv, arr, delimiter=",", header="t,pos_x,pos_y,pos_z,set_x,set_y,set_z", comments="")
    print("Saved CSV :", os.path.abspath(csv))

    # quick metrics
    if len(pos)>1 and len(sp)==len(pos):
        v = pos[1:] - pos[:-1]
        to_sp = sp[:-1] - pos[:-1]
        dot = np.sum(v*to_sp, axis=1)
        toward = (dot>0).mean()
        gap = np.linalg.norm(to_sp, axis=1) + 1e-9
        prog = np.linalg.norm(v, axis=1)/gap
        print(f"Toward-setpoint steps: {toward*100:.1f}% | Median progress/tick: {np.median(prog):.2f}")
    else:
        print("Note: setpoint series empty or mismatched. Ensure env sets _last_pid_target and exports it in info.")

if __name__ == "__main__":
    main()