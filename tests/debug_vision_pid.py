import numpy as np, argparse, math, time
import gymnasium as gym
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

SUCCESS_EPS = 0.30
TARGET = np.array([0.0, 0.0, 1.5], dtype=np.float32)
SCALE_RPM = 20000.0  # normalized action -> env

def make_env(gui=False, obstacles=False, seed=123):
    env = VisionAviary(gui=gui, record=False, obstacles=obstacles)
    env.reset(seed=seed)
    return env

def get_state(base):
    s = base._getDroneStateVector(0)
    # κοινές διατάξεις: [x,y,z, qx,qy,qz,qw, roll,pitch,yaw, ...] ή [x,y,z, ... roll,pitch,yaw ...]
    # θα προσπαθήσουμε να πάρουμε roll/pitch από τις 7..9, αλλιώς 0.
    roll = float(s[7]) if len(s) > 8 else 0.0
    pitch = float(s[8]) if len(s) > 9 else 0.0
    pos = s[0:3].astype(np.float32)
    return pos, roll, pitch

def pid_to_action(env, target_pos, alt_boost=False, boost_max=1.35):
    base = env.unwrapped
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
    if alt_boost:
        # εκτίμηση tilt από roll/pitch τρέχουσας στάσης
        _, roll, pitch = get_state(base)
        tilt = math.sqrt(roll*roll + pitch*pitch)  # rad
        boost = 1.0 / max(0.01, math.cos(min(abs(tilt), math.radians(50))))  # 1/cos(tilt)
        boost = float(np.clip(boost, 1.0, boost_max))
        rpms = rpms * boost
    act = np.clip(rpms / SCALE_RPM, 0.0, 1.0).astype(np.float32)
    clip_frac = float(np.mean(act >= 0.999))
    return act, rpms, clip_frac

def run(gui=False, steps=300, alt_boost=False):
    env = make_env(gui=gui, obstacles=False, seed=777)
    base = env.unwrapped
    obs, info = env.reset()

    z0 = float(base._getDroneStateVector(0)[2])
    min_dist = 1e9
    print("t | z     | dist   | roll  | pitch | tilt  | rpm_min rpm_mean rpm_max | clip")
    for t in range(steps):
        pos, roll, pitch = get_state(base)
        dist = float(np.linalg.norm(pos - TARGET))
        min_dist = min(min_dist, dist)

        act, rpms, clip = pid_to_action(env, TARGET, alt_boost=alt_boost)
        rpm_min, rpm_mean, rpm_max = float(rpms.min()), float(rpms.mean()), float(rpms.max())
        tilt = math.degrees(math.sqrt(roll*roll + pitch*pitch))

        print(f"{t:3d} | {pos[2]:5.2f} | {dist:6.3f} | {roll:+5.2f} | {pitch:+5.2f} | {tilt:5.1f} | "
              f"{rpm_min:6.0f} {rpm_mean:7.0f} {rpm_max:6.0f} | {clip:4.2f}")

        obs, rew, term, trunc, _ = env.step(act)
        if dist < SUCCESS_EPS or term or trunc:
            break

    zN = float(base._getDroneStateVector(0)[2])
    env.close()
    print(f"\nRESULT  reached={min_dist<SUCCESS_EPS}  min_dist={min_dist:.3f}  Δz={zN - z0:+.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--boost", action="store_true", help="enable altitude compensation (1/cos tilt)")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    run(gui=args.gui, steps=args.steps, alt_boost=args.boost)