# preview_map.py — Level preview for VisionAviary (robust imports)
import time
import argparse
import numpy as np
import pybullet as p

# --- robust enums import (supports multiple forks) ---
ObservationType = None
ActionType = None

# Try common locations first
try:
    from gym_pybullet_drones.utils.enums import ObservationType as _OT, ActionType as _AT
    ObservationType, ActionType = _OT, _AT
except Exception:
    try:
        from gym_pybullet_drones.envs.BaseAviary import ObservationType as _OT, ActionType as _AT
        ObservationType, ActionType = _OT, _AT
    except Exception:
        # Fallback shims
        from enum import Enum
        class _OT(Enum):
            KIN = "kin"
            RGB = "rgb"
        class _AT(Enum):
            PID = "pid"
        ObservationType, ActionType = _OT, _AT

from gym_pybullet_drones.envs.VisionAviary import VisionAviary


def _coerce_obs(obs_mode):
    """
    Return the appropriate obs object depending on whether ObservationType
    is a callable factory or an Enum. Falls back to raw string.
    """
    mode = obs_mode.lower()
    # Try callable (factory-style) — e.g., ObservationType('rgb')
    try:
        return ObservationType(mode)  # will work if it's a callable
    except Exception:
        pass
    # Try Enum style
    try:
        return ObservationType.RGB if mode == "rgb" else ObservationType.KIN
    except Exception:
        # Last resort: raw string
        return mode


def _coerce_act(act_mode="pid"):
    mode = act_mode.lower()
    # Try callable
    try:
        return ActionType(mode)
    except Exception:
        pass
    # Try Enum
    try:
        return ActionType.PID
    except Exception:
        return mode


def _infer_goal_from_start(start_xyz, keep_goal_z=True):
    g = np.array(start_xyz, dtype=np.float32).copy()
    g[0] += 5.0  # "in front" = +X
    if keep_goal_z:
        g[2] = start_xyz[2]
    return g


def _ensure_goal(env):
    """Make sure env.goal exists; if not, infer it from INIT_XYZS or current pose."""
    if getattr(env, "goal", None) is None:
        if hasattr(env, "INIT_XYZS"):
            start = np.array(env.INIT_XYZS[0], dtype=np.float32)
        else:
            start = np.array(env._getDroneStateVector(0)[0:3], dtype=np.float32)
        env.goal = _infer_goal_from_start(
            start, getattr(env, "keep_goal_z_equal_spawn", True)
        ).astype(np.float32)


def _draw_marker(pos, color=(1, 1, 1, 0.7), radius=0.08, client_id=0):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client_id)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(map(float, pos)), physicsClientId=client_id)


def _draw_line(a, b, color=(1, 1, 1), width=2.0, client_id=0, lifeTime=0):
    p.addUserDebugLine(list(map(float, a)), list(map(float, b)), lineColorRGB=color, lineWidth=width, lifeTime=lifeTime, physicsClientId=client_id)


def _draw_text(text, pos, color=(1, 1, 1), size=1.2, client_id=0, lifeTime=0):
    p.addUserDebugText(text, list(map(float, pos)), textColorRGB=color, textSize=size, lifeTime=lifeTime, physicsClientId=client_id)


def show(difficulty: int, seconds: float = 8.0, seed: int = 42, obs_mode="rgb"):
    """
    Preview VisionAviary level geometry.

    difficulty:
      1: goal 5m ahead, no obstacles
      2: goal 5m ahead, single cylinder midway
      3: three hoops along the path
    """
    obs = _coerce_obs(obs_mode)
    act = _coerce_act("pid")

    env = VisionAviary(
        gui=True,
        obs=obs,
        act=act,
        ctrl_freq=24,
        difficulty=difficulty,
        record=False,
        random_start=True,
    )

    # Let env finalize spawn/goal & obstacles
    env.reset(seed=seed)
    _ensure_goal(env)
    
    client = env.CLIENT
    start = np.array(env.INIT_XYZS[0], dtype=np.float32)
    goal = np.array(env.goal, dtype=np.float32)

    # Visual aids
    _draw_marker(start, color=(0.2, 0.4, 1.0, 0.8), radius=0.10, client_id=client)  # blue: spawn
    _draw_marker(goal,  color=(0.1, 0.9, 0.1, 0.8), radius=0.12, client_id=client)  # green: goal
    _draw_line(start, goal, color=(1, 1, 1), width=2.0, client_id=client)

    _draw_text("SPAWN", start + np.array([0, 0, 0.25]), color=(0.6, 0.8, 1.0), size=1.4, client_id=client)
    _draw_text("GOAL",  goal  + np.array([0, 0, 0.25]), color=(0.6, 1.0, 0.6), size=1.4, client_id=client)

    level_label = {
        0: "Level 0 (open)",
        1: "Level 1 (no obstacles)",
        2: "Level 2 (cylinder)",
        3: "Level 3 (hoops)",
    }.get(int(difficulty), f"Level {difficulty}")
    _draw_text(level_label, start + np.array([0.0, -0.7, 0.6]), color=(1, 1, 0.6), size=1.5, client_id=client)

    # Keep GUI up
    t0 = time.time()
    while time.time() - t0 < max(0.0, seconds):
        env.render()
        time.sleep(1 / 60)

    env.close()

    env.export_current_level_to_obj("level_export.obj")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview VisionAviary level geometry.")
    parser.add_argument("--difficulty", type=int, default=2, help="1=no obstacles, 2=cylinder, 3=hoops")
    parser.add_argument("--seconds", type=float, default=8.0, help="How long to display the scene")
    parser.add_argument("--seed", type=int, default=42, help="Reset seed")
    parser.add_argument("--obs", type=str, default="rgb", choices=["rgb", "kin"], help="Observation preview mode")
    args = parser.parse_args()

    show(args.difficulty, args.seconds, args.seed, obs_mode=args.obs)
