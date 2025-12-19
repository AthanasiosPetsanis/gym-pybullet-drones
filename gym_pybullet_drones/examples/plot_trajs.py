import matplotlib.pyplot as plt
import numpy as np
import glob, os

run_dir  = r"results\save-12.16.2025_22.47.59"
traj_dir = os.path.join(run_dir, "trajs")
files = sorted(glob.glob(os.path.join(traj_dir, "traj_env0_ep*.npy")))

# --- Τροχιές ---
for f in files:
     traj = np.load(f)          # (T, 3)
     plt.plot(traj[:,0], traj[:,1], alpha=0.05)
# traj = np.load(files[-2])          # (T, 3)
# plt.plot(traj[:,0], traj[:,1], alpha=0.05)

# --- Δίσκος εκκίνησης (radius 1.2 γύρω από (0,0)) ---
theta = np.linspace(0, 2*np.pi, 200)
r = 1.2
plt.plot(r*np.cos(theta), r*np.sin(theta), "k--", linewidth=1, label="start radius")

# --- Room envelope για difficulty=3 (6m x 3m) ---
start_x, start_y = 0.0, 0.0
ROOM_L, ROOM_W = 6.0, 3.0
x0, x1 = start_x, start_x + ROOM_L
y0, y1 = start_y - ROOM_W/2, start_y + ROOM_W/2
plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], "r-", linewidth=1.5, label="room")

# --- Goal (στην ευθεία +X) ---
goal_x, goal_y = 5.0, 0.0
plt.scatter([goal_x], [goal_y], marker="*", s=80, c="gold", edgecolors="k", label="goal")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.legend()
plt.title("Trajectories + track (env 0)")
plt.show()