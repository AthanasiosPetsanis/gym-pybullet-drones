import numpy as np, time
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

SCALE = 21700.0
EPS = 0.10        # +10% στο συγκεκριμένο μοτέρ (normalized)
SETTLE = 30
PUSH = 40

def main(gui=False):
    env = VisionAviary(gui=gui, record=False, obstacles=False)
    base = env.unwrapped
    env.reset()

    # hover approx
    hover_rpm = float(getattr(base, "HOVER_RPM", 14468.0))
    a_hover = np.clip(np.ones(4, dtype=np.float32) * (hover_rpm / SCALE), 0.0, 1.0)

    for m in range(4):
        env.reset()
        # settle
        for _ in range(SETTLE):
            env.step(a_hover)
        p0 = base._getDroneStateVector(0)[0:3].copy()

        a = a_hover.copy()
        a[m] = float(np.clip(a[m] + EPS, 0.0, 1.0))
        for _ in range(PUSH):
            env.step(a)

        p1 = base._getDroneStateVector(0)[0:3].copy()
        d = p1 - p0
        print(f"motor {m}: Δx={d[0]:+.3f}  Δy={d[1]:+.3f}  Δz={d[2]:+.3f}")

    env.close()

if __name__ == "__main__":
    main(gui=False)