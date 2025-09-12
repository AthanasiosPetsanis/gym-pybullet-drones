import os, glob, time, argparse
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

def newest_run_dir(root="results"):
    runs = sorted(glob.glob(os.path.join(root, "save-*")), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("No results/save-* folder found.")
    return runs[-1]

def model_path(run_dir, prefer="best"):
    best  = os.path.join(run_dir, "best_model.zip")
    final = os.path.join(run_dir, "final_model.zip")
    if prefer == "best" and os.path.isfile(best):
        return best
    return best if os.path.isfile(best) else final

def guess_obs_kind_from_policy(pol_obs_space):
    """Heuristic: uint8 images -> 'rgb'; otherwise vector -> 'kin' (stacked or not)."""
    if hasattr(pol_obs_space, "dtype") and str(pol_obs_space.dtype) == "uint8":
        return "rgb"
    # shapes like (N,) or (4, N) -> KIN (stack=1 or 4)
    return "kin"

def make_env(obs_kind: str, difficulty: int, gui: bool, record: bool):
    return VisionAviary(obs=ObservationType(obs_kind),
                        act=ActionType('pid'),
                        ctrl_freq=24,
                        difficulty=int(difficulty),
                        gui=gui, record=record)

def needs_stack(policy_shape, env_shape):
    """Detect if the loaded policy expects 4-frame stacks along leading axis."""
    if isinstance(policy_shape, tuple) and isinstance(env_shape, tuple):
        # Example: policy (4, 48) vs env (48,)  OR  policy (4, 3, 84, 84) vs env (3, 84, 84)
        return len(policy_shape) == len(env_shape) + 1 and policy_shape[0] == 4 and policy_shape[1:] == env_shape
    return False

def run_episode(model, env, deterministic=True, realtime=False):
    """
    Works for KIN/RGB, with or without frame-stacking.
    Auto-enables stacking on SB3's shape error, and squeezes (1, D) -> (D).
    """
    def squeeze_obs(o):
        # KIN often comes as (1, obs_dim); make it (obs_dim,)
        if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[0] == 1:
            return o.squeeze(0)
        return o

    obs, _ = env.reset()
    obs = squeeze_obs(obs)

    done = trunc = False
    ep_ret, steps = 0.0, 0
    actions = []
    start = time.time()

    use_stack = False
    buf = None

    def current_obs():
        if not use_stack:
            return obs
        # Always squeeze before stacking so shapes are (4, obs_dim) or (4, C, H, W)
        return np.stack([squeeze_obs(x) for x in list(buf)], axis=0)

    while not (done or trunc):
        try:
            inp = current_obs()
            action, _ = model.predict(inp, deterministic=deterministic)
        except ValueError as e:
            # If policy expects stacked input (e.g., "please use (4, 48) or (n_env, 4, 48)")
            if "(4," in str(e) and not use_stack:
                use_stack = True
                buf = deque([obs.copy() for _ in range(4)], maxlen=4)
                inp = current_obs()
                action, _ = model.predict(inp, deterministic=deterministic)
                print("[INFO] Policy expects frame stacking (n_stack=4) — enabled in evaluator.")
            else:
                raise

        actions.append(np.asarray(action).reshape(-1, 3))
        obs, r, done, trunc, info = env.step(action)
        obs = squeeze_obs(obs)
        if use_stack:
            buf.append(obs)
        ep_ret += float(r); steps += 1

        if realtime:
            env.render()
            sync(env.step_counter, start, env.CTRL_TIMESTEP)

    acts = np.concatenate(actions, axis=0) if actions else np.zeros((0, 3))
    success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
    return ep_ret, steps, success, acts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None, help="results/save-* folder (default: newest)")
    ap.add_argument("--prefer",  type=str, default="best", choices=["best","final"])
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stochastic", action="store_true", help="sample actions (else deterministic)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="real-time stepping when GUI is on")
    ap.add_argument("--demo_seconds", type=float, default=6.0, help="hover at end so you can look at the map")
    ap.add_argument("--record_video", action="store_true")
    ap.add_argument("--print_actions", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir or newest_run_dir()
    mp = model_path(run_dir, prefer=args.prefer)
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"No model zip found in {run_dir}")
    print(f"[INFO] Using run:   {run_dir}")
    print(f"[INFO] Loading:     {mp}")

    model = PPO.load(mp, device="cpu")
    pol_space = model.observation_space  # saved with the policy
    obs_kind  = guess_obs_kind_from_policy(pol_space)
    print(f"[INFO] Detected policy obs space: {pol_space} -> using obs='{obs_kind}'")

    # Build env accordingly
    env = make_env(obs_kind, args.difficulty, gui=args.gui, record=args.record_video)
    print("[INFO] Env action space:", env.action_space)
    print("[INFO] Env obs space   :", env.observation_space)

    # Decide whether we must stack 4 obs to match the policy
    stack = needs_stack(tuple(pol_space.shape), tuple(env.observation_space.shape))
    if stack:
        print("[INFO] Policy expects stacked observations (n_stack=4) — stacking in evaluator.")
    else:
        print("[INFO] No frame stacking needed.")

    # Rollouts
    rets, lens, succs = [], [], []
    for ep in range(args.episodes):
        ep_ret, steps, success, acts = run_episode(
            model, env,
            deterministic=not args.stochastic,
            realtime=args.gui and args.realtime
        )
        rets.append(ep_ret); lens.append(steps); succs.append(success)
        if args.print_actions and acts.size:
            print(f"[ACTIONS ep{ep}] min {acts.min(axis=0)} | max {acts.max(axis=0)} "
                  f"| mean {acts.mean(axis=0)} | std {acts.std(axis=0)}")
        print(f"[EP {ep+1}/{args.episodes}] return={ep_ret:.2f} | steps={steps} | success={int(success)}")

    print("\n=== Evaluation summary ===")
    print(f"episodes       : {len(rets)}")
    print(f"mean return    : {np.mean(rets):.2f} ± {np.std(rets):.2f}")
    print(f"mean ep length : {np.mean(lens):.2f}")
    print(f"success rate   : {100.0*np.mean(succs):.1f}%")

    # small hover so the GUI stays up for inspection
    if args.gui and args.demo_seconds > 0:
        print(f"\n[INFO] Demo hover for {args.demo_seconds}s…")
        obs, _ = env.reset()
        start = time.time()
        while time.time() - start < args.demo_seconds:
            pos = env._getDroneStateVector(0)[0:3]
            action = pos[None, :].astype(np.float32)  # PID setpoint (1,3)
            obs, _, _, _, _ = env.step(action)
            env.render()
            sync(env.step_counter, start, env.CTRL_TIMESTEP)
    env.close()

if __name__ == "__main__":
    main()