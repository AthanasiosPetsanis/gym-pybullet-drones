import numpy as np
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from stable_baselines3.common.monitor import Monitor

# ⬇️ import τον wrapper ΑΠΟ το train_ppo_vision.py σου (ίδιος κώδικας=ίδια συμπεριφορά)
from train_ppo_vision import MacroPIDWaypointWrapper

def main(gui=False):
    env = VisionAviary(gui=gui, record=False, obstacles=False)
    env = Monitor(env)

    # ΧΡΗΣΙΜΟ: action_space μετά το wrapper πρέπει να είναι (3,)  (dx,dy,dz)
    env = MacroPIDWaypointWrapper(
        env,
        delta_limit=2.0,
        subgoal_eps=0.20,
        max_pid_steps=12,
        boost_altitude=True,
        boost_max=1.25,
        scale_rpm=20000.0,   # ίδιο με το train
    )
    print("Top-level action_space (should be 3D):", env.action_space)

    obs, info = env.reset()
    base = env.unwrapped

    # 1) Ζήτα "πάνω" μερικά RL-βήματα (ο wrapper θα κάνει macro PID steps)
    for _ in range(3):
        obs, r, term, trunc, _ = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # 2) Κυνήγα τον στόχο [0,0,1.5] με clipped deltas
    target = np.array([0.0, 0.0, 1.5], dtype=np.float32)
    for t in range(30):
        pos = base._getDroneStateVector(0)[0:3]
        delta = np.clip(target - pos, -1.0, 1.0).astype(np.float32)  # να ταιριάζει με το wrapper
        obs, r, term, trunc, _ = env.step(delta)
        pos = base._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(target - pos))
        dist_xy = float(np.linalg.norm(pos[:2] - target[:2]))
        print(f"step {t:02d}: (x,y,z)=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
        f"dist_xy={dist_xy:.3f}  dist={dist:.3f}")
        if dist < 0.30 or term or trunc:
            print("RESULT: reached" if dist < 0.30 else "RESULT: terminated")
            break

    env.close()

if __name__ == "__main__":
    # Βάλε gui=True αν θέλεις να το δεις οπτικά
    main(gui=True)