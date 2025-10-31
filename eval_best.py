import os, glob, argparse, time
from collections import deque
import numpy as np

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


# ----------------------- utilities -----------------------

def newest_run_dir(root="results"):
    runs = sorted(glob.glob(os.path.join(root, "save-*")), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("No results/save-* directory found.")
    return runs[-1]

def pick_model(run_dir, prefer="final"):
    best  = os.path.join(run_dir, "best_model.zip")
    final = os.path.join(run_dir, "final_model.zip")
    if prefer == "best" and os.path.isfile(best):
        return best
    return best if os.path.isfile(best) else final

def coerce_obs_single_env(obs: np.ndarray, policy_shape: tuple, buf: deque | None):
    """
    Coerce a single-env observation to exactly match the policy observation shape.
    Handles three cases:
      - exact match                 e.g., env (48,)         == policy (48,)
      - batch/stack=1               e.g., env (48,)         -> policy (1, 48)
      - frame stack=4               e.g., env (48,) x4      -> policy (4, 48)
      - same logic for RGB          e.g., env (3,84,84)     -> (1,3,84,84) or (4,3,84,84)

    If buf is provided (deque), it already contains the most recent frames.
    """
    eshape = tuple(obs.shape)
    pshape = tuple(policy_shape)

    # exact match
    if eshape == pshape:
        return obs

    # need one leading axis (batch/stack=1)
    if len(pshape) == len(eshape) + 1 and pshape[0] == 1 and pshape[1:] == eshape:
        return np.expand_dims(obs, axis=0)

    # need stack=4 along leading axis
    if len(pshape) == len(eshape) + 1 and pshape[0] == 4 and pshape[1:] == eshape:
        assert buf is not None and len(buf) == 4, "internal: stack buffer not ready"
        return np.stack(list(buf), axis=0)

    # fallback: try to broadcast with single leading axis
    if len(pshape) == len(eshape) + 1 and pshape[1:] == eshape:
        # assume stack=1
        return np.expand_dims(obs, axis=0)

    raise ValueError(f"Cannot coerce env obs shape {eshape} to policy shape {pshape}")

def make_env(obs_kind: str, difficulty: int, gui: bool, record: bool):
    return VisionAviary(
        obs=ObservationType(obs_kind),
        act=ActionType('pid'),
        ctrl_freq=24,
        difficulty=int(difficulty),
        gui=gui,
        record=record,
        random_start=True,
        start_center_xy=(0.0, 0.0),
        start_radius=1.2,
        start_z_range=(0.75, 0.95),
        keep_goal_z_equal_spawn=True,
    )

def guess_obs_kind_from_shapes(policy_shape: tuple):
    # If last dims look like (3,H,W), assume RGB; else KIN
    if len(policy_shape) >= 3 and policy_shape[-3] == 3:
        return "rgb"
    return "kin"


# ----------------------- main eval loop -----------------------

def run_episode(model, env, policy_shape: tuple, deterministic=True, realtime=False):
    """Run one episode; coerce obs shape to policy_shape; auto-handle stack=1 or 4."""
    # prepare buffer if policy expects stack=4
    stack_len = 0
    if len(policy_shape) >= 2:
        # if leading axis is 4, assume frame stack
        if policy_shape[0] == 4:
            stack_len = 4
        elif policy_shape[0] == 1:
            stack_len = 1

    buf = deque(maxlen=max(stack_len, 1))

    def squeeze_batch1(o):
        # some envs return (1,·); drop that leading axis if present
        if isinstance(o, np.ndarray) and o.ndim >= 2 and o.shape[0] == 1 and len(policy_shape) == o.ndim - 1:
            return o.squeeze(0)
        return o

    obs, _ = env.reset()
    obs = squeeze_batch1(obs)

    # prime buffer
    if stack_len > 0:
        for _ in range(stack_len):
            buf.append(obs.copy())

    done = trunc = False
    ep_ret, steps = 0.0, 0
    actions = []
    start = time.time()

    while not (done or trunc):
        # build input matching policy shape
        inp = coerce_obs_single_env(buf[-1] if stack_len else obs, policy_shape, buf if stack_len == 4 else None)

        action, _ = model.predict(inp, deterministic=deterministic)
        actions.append(np.asarray(action).reshape(-1, 3))

        obs, r, done, trunc, info = env.step(action)
        obs = squeeze_batch1(obs)
        if stack_len:
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
    ap.add_argument("--prefer",  type=str, default="final", choices=["best","final"], help="load best_model or final_model")
    ap.add_argument("--difficulty", type=int, default=2, help="track level; 2 = cylinder")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--record_video", action="store_true")
    ap.add_argument("--print_actions", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir or newest_run_dir()
    mp = pick_model(run_dir, prefer=args.prefer)
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"No model zip found in {run_dir}")
    print(f"[INFO] Using run:   {run_dir}")
    print(f"[INFO] Loading:     {mp}")

    # load on CPU (good for MLP/KIN)
    model = PPO.load(mp, device="cpu")
    policy_shape = tuple(model.observation_space.shape)
    obs_kind = guess_obs_kind_from_shapes(policy_shape)
    print(f"[INFO] Policy obs shape: {policy_shape} -> obs_kind='{obs_kind}'")
    print(f"[INFO] Level (difficulty): {args.difficulty}")

    env = make_env(obs_kind, args.difficulty, gui=args.gui, record=args.record_video)
    print("[INFO] Env action space:", env.action_space)
    print("[INFO] Env obs space   :", env.observation_space)

    # quick note about stacking expected by policy
    if len(policy_shape) > 0 and policy_shape[0] in (1, 4) and len(policy_shape) == len(env.observation_space.shape) + 1:
        print(f"[INFO] Policy expects leading stack/batch axis = {policy_shape[0]} → will coerce input shape accordingly.")
    else:
        print("[INFO] Policy/env shapes appear aligned (no coercion beyond batch handling).")

    # run episodes
    rets, lens, succs = [], [], []
    for ep in range(args.episodes):
        ep_ret, steps, success, acts = run_episode(
            model, env, policy_shape,
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

    env.close()


if __name__ == "__main__":
    main()