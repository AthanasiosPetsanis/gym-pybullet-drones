import numpy as np
from gymnasium import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class VisionAviary(BaseRLAviary):
    def __init__(
        self,
        gui=False,
        record=False,
        act=None,
        obs=None,
        obstacles=True,
        difficulty: int = 0,
        initial_xyzs=None,
        initial_rpys=None,
        random_start: bool = True,
        start_center_xy=(0.0, 0.0),
        start_radius: float = 1.2,
        start_z_range=(0.75, 0.95),
        keep_goal_z_equal_spawn: bool = True,
        ceiling_margin: float = 0.6,
        pyb_freq: int = 240,
        ctrl_freq: int = 24,
    ):
        self.difficulty = int(difficulty)
        self._obstacles_enabled = obstacles
        self._obstacle_ids = []      # track bodies we spawn
        self._goal_id = None
        self.z_max = 1.4           # hard ceiling height (m)
        self.use_ceiling = True    # turn the invisible ceiling on/off
        self.success_thresh = 0.4     # meters
        self.hold_steps_req = 10       # must stay inside for this many steps to count success
        self._hold_count = 0
        self.progress_w     = 15.0          # weight for progress (prev_dist - dist)
        self.dist_w         = 0.15          # constant pull toward the goal each step
        self.step_cost      = 0.02          # tiny time penalty per step
        self.success_bonus  = 300.0          # one-time bonus for reaching the goal
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=initial_xyzs,
            physics=Physics.PYB,
            gui=gui,
            record=record,
            act=act,
            obs=obs,
            initial_rpys=initial_rpys,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
        )

        # PID controllers
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.NUM_DRONES)]

        # default spawn if none provided
        if initial_xyzs is None:
            self.INIT_XYZS = np.array([[0.0, 0.0, 1.2]], dtype=np.float32)

        # camera
        self.IMG_RES = (84, 84)  # (W, H)
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL

        # constructor-time goal (overwritten on reset)
        self.goal = np.array([2.0, 4.0, 1.0], dtype=np.float32)
        if self._obstacles_enabled:
            self._addObstacles()

        # bookkeeping
        self._prev_pos = None
        self._last_distance = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._gates = []
        self._gates_passed = 0

        # observation space for RGB
        if self.OBS_TYPE == ObservationType.RGB:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(3, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8
            )

    # ---------- RL API ----------
    def reset(self, seed=None, options=None):
        rng = np.random.default_rng()

        if self.random_start:
            # uniform in disk
            u = rng.uniform()
            r = np.sqrt(u) * self.start_radius
            th = rng.uniform(0, 2 * np.pi)
            cx, cy = self.start_center_xy
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)
            z = rng.uniform(*self.start_z_range)
            start = np.array([x, y, z], dtype=np.float32)
        else:
            start = np.array(self.INIT_XYZS[0], dtype=np.float32)

        # Goal: 5 m in +X, keep Z equal to spawn if requested
        goal_vec = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        goal_xyz = start + goal_vec
        if self.keep_goal_z_equal_spawn:
            goal_xyz[2] = start[2]

        self.goal = goal_xyz.astype(np.float32)
        self.INIT_XYZS = start.reshape(1, 3)

        # ceiling safety margin
        if getattr(self, "use_ceiling", True):
            self.z_max = float(max(self.z_max, start[2] + self.ceiling_margin))

        # (re)build geometry for this level
        self._build_level_geometry(start, self.goal)

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.KIN:
            return super()._computeObs()
        img, _, _ = self._getDroneImages(0)
        return img

    def _computeReward(self):
        pos = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))

        progress = 0.0 if self._last_distance is None else (self._last_distance - dist)
        self._last_distance = dist

        reward = self.progress_w * progress
        reward += -self.dist_w * dist
        reward += -self.step_cost

        if dist < self.success_thresh:
            reward += self.success_bonus

        if dist < 0.8:
            reward += -0.5 * abs(pos[2] - self.goal[2])

        if self._isCollision(0):
            reward -= 100.0

        return reward

    def _computeTerminated(self):
        pos = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))
        all_gates_ok = True
        if getattr(self, "_gates", None) and self.difficulty == 3:
            all_gates_ok = (self._gates_passed >= len(self._gates))
        if dist < self.success_thresh and all_gates_ok:
            return True
        if self._isCollision(0):
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter >= 1300

    def _computeInfo(self):
        """Also updates gate-pass counter for Level 3."""
        pos = self._getDroneStateVector(0)[0:3]

        # Level-3 gate pass-through (run every step, not during obstacle build)
        if getattr(self, "_gates", None) and self.difficulty == 3 and self._gates_passed < len(self._gates):
            x_plane, y_center, z_hole, R_hole = self._gates[self._gates_passed]
            near_plane = abs(float(pos[0] - x_plane)) <= 0.18       # close to wall plane in X
            dy, dz = float(pos[1] - y_center), float(pos[2] - z_hole)
            in_circle = (dy * dy + dz * dz) <= (R_hole - 0.05) ** 2 # inside the opening
            if near_plane and in_circle:
                self._gates_passed += 1

        dist = float(np.linalg.norm(self.goal - pos))
        return {"distance_to_goal": dist, "is_success": dist < self.success_thresh}

    # ---------- Helpers ----------
    def _getDroneImages(self, drone_id=0, segmentation=False):
        pos = self._getDroneStateVector(drone_id)[:3]
        quat = self._getDroneStateVector(drone_id)[3:7]
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = R @ np.array([1, 0, 0])

        cam_pos = pos + np.array([0.0, 0.0, 0.2])
        tgt_pos = pos + forward

        view = p.computeViewMatrix(cam_pos.tolist(), tgt_pos.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.IMG_RES[0]) / self.IMG_RES[1], nearVal=0.1, farVal=10.0
        )
        w, h, rgba, _, _ = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=self.RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
        return rgb, None, None

    def _isCollision(self, drone_id=0):
        cps = p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)
        return len(cps) > 0

    def _addObstacles(self):
        """
        Level geometry:
          1: goal 5 m ahead, no obstacles
          2: one cylinder midway
          3: closed room + three compact walls (with a circular hole at varying heights)
        Safe during __init__: derives a default goal from spawn if needed.
        """
        CLIENT = self.CLIENT

        # ----- ensure bookkeeping
        if not hasattr(self, "_obstacle_ids"):
            self._obstacle_ids = []
        if not hasattr(self, "_goal_id"):
            self._goal_id = None

        # ----- cleanup previous
        for bid in list(self._obstacle_ids):
            try:
                p.removeBody(bid, physicsClientId=CLIENT)
            except Exception:
                pass
        self._obstacle_ids.clear()

        if self._goal_id is not None:
            try:
                p.removeBody(self._goal_id, physicsClientId=CLIENT)
            except Exception:
                pass
            self._goal_id = None

        # ----- derive start & goal safely
        if hasattr(self, "INIT_XYZS") and len(self.INIT_XYZS) > 0:
            start = np.array(self.INIT_XYZS[0], dtype=np.float32)
        else:
            start = np.array([0.0, 0.0, 0.8], dtype=np.float32)

        if not hasattr(self, "goal") or self.goal is None:
            g = start.copy()
            g[0] += 5.0
            if getattr(self, "keep_goal_z_equal_spawn", True):
                g[2] = start[2]
            self.goal = g.astype(np.float32)

        goal = np.array(self.goal, dtype=np.float32)

        # ----- helpers
        def _track(bid):
            if bid is not None:
                self._obstacle_ids.append(bid)
            return bid

        def add_cylinder(x, y, radius=0.30, height=1.6, rgba=(0.85, 0.35, 0.35, 0.95)):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=CLIENT)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba, physicsClientId=CLIENT)
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(x), float(y), float(height / 2.0)],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=CLIENT,
            )
            return _track(bid)

        def add_box(color_rgba, center_xyz, half_extents):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=CLIENT)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color_rgba, physicsClientId=CLIENT)
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=CLIENT,
            )
            return _track(bid)

        def add_panel_with_round_hole(
            x_wall,
            y_center,
            z_center,
            room_W,
            room_H,
            thickness=0.08,
            hole_radius=0.35,
            border=0.12,
            color=(0.2, 0.6, 1.0, 0.9),
        ):
            """Four boxes that leave a circular opening of radius `hole_radius`."""
            W, H = float(room_W), float(room_H)
            T, R, B = float(thickness), float(hole_radius), float(border)

            # μισά μεγέθη για τα 4 τεμάχια
            side_half_y = max((W/2 - R - B), 0.05) / 2.0
            top_half_z  = max((H/2 - R - B), 0.05) / 2.0

            # LEFT / RIGHT slabs
            cy_left  = y_center - (W/2 + R + B)/2.0
            add_box(color, [x_wall, cy_left,  z_center], [T/2.0, side_half_y, H/2.0])
            cy_right = y_center + (W/2 + R + B)/2.0
            add_box(color, [x_wall, cy_right, z_center], [T/2.0, side_half_y, H/2.0])

            # TOP / BOTTOM slabs
            cz_top = z_center + (H/2 + R + B)/2.0
            add_box(color, [x_wall, y_center, cz_top], [T/2.0, W/2.0, top_half_z])
            cz_bot = z_center - (H/2 + R + B)/2.0
            add_box(color, [x_wall, y_center, cz_bot], [T/2.0, W/2.0, top_half_z])

            # Remember gate (for pass-through check)
            self._gates.append((float(x_wall), float(y_center), float(z_center), float(R)))

        def add_room_envelope(room_center, room_L=6.0, room_W=3.0, room_H=2.2, wall_T=0.10, color=(0.7, 0.7, 0.7, 0.9)):
            """Four perimeter walls + ceiling (no floor)."""
            cx, cy, cz = float(room_center[0]), float(room_center[1]), float(room_center[2])
            L, W, H, T = float(room_L), float(room_W), float(room_H), float(wall_T)
            # back/front
            add_box(color, [cx - L / 2.0, cy, cz], [T / 2.0, W / 2.0, H / 2.0])
            add_box(color, [cx + L / 2.0, cy, cz], [T / 2.0, W / 2.0, H / 2.0])
            # left/right
            add_box(color, [cx, cy - W / 2.0, cz], [L / 2.0, T / 2.0, H / 2.0])
            add_box(color, [cx, cy + W / 2.0, cz], [L / 2.0, T / 2.0, H / 2.0])
            # ceiling
            add_box(color, [cx, cy, H], [L / 2.0, W / 2.0, T / 2.0])

        # ----- goal marker
        try:
            self._goal_id = p.loadURDF(
                "cube.urdf",
                goal.tolist(),
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                globalScaling=0.15,
                physicsClientId=CLIENT,
            )
            p.changeVisualShape(self._goal_id, -1, rgbaColor=[0, 1, 0, 1], physicsClientId=CLIENT)
        except Exception:
            self._goal_id = None

        # ----- levels
        self._gates = []
        self._gates_passed = 0
        d = int(getattr(self, "difficulty", 0))

        if d == 0:
            pass

        elif d == 1:
            pass

        elif d == 2:
            mid = 0.5 * (start + goal)
            add_cylinder(x=mid[0], y=mid[1], radius=0.30, height=1.6)

        elif d == 3:
            # closed room + three compact walls with circular holes (various heights)
            ROOM_L, ROOM_W, ROOM_H, ROOM_T = 6.0, 3.0, 2.2, 0.10
            room_cx = float(start[0]) + ROOM_L / 2.0
            room_cy = float(start[1])
            room_cz = ROOM_H / 2.0
            add_room_envelope([room_cx, room_cy, room_cz], ROOM_L, ROOM_W, ROOM_H, ROOM_T, color=(0.7, 0.7, 0.7, 0.9))

            x_back = room_cx - ROOM_L / 2.0 + 1.0
            x_mid = room_cx
            x_front = room_cx + ROOM_L / 2.0 - 1.2
            gate_xs = [x_back, x_mid, x_front]

            base_z = float(start[2])
            hole_zs = [base_z + 0.45, base_z - 0.35, base_z + 0.25]
            hole_R, border, Tpanel = 0.30, 0.08, 0.08

            for xw, zh in zip(gate_xs, hole_zs):
                add_panel_with_round_hole(
                    x_wall=xw,
                    y_center=room_cy,
                    z_center=zh,
                    room_W=ROOM_W,
                    room_H=ROOM_H,
                    thickness=Tpanel,
                    hole_radius=hole_R,
                    border=border,
                    color=(0.2, 0.6, 1.0, 0.9),
                )

            # move goal just beyond last wall at height of last hole
            try:
                self.goal = np.array([x_front + 0.8, room_cy, hole_zs[-1]], dtype=np.float32)
                if self._goal_id is not None:
                    p.resetBasePositionAndOrientation(
                        self._goal_id, self.goal.tolist(), p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=CLIENT
                    )
            except Exception:
                pass

        # ----- invisible ceiling only for non-room levels
        if d != 3 and getattr(self, "use_ceiling", True) and hasattr(self, "z_max"):
            try:
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.02], physicsClientId=CLIENT)
                ceiling = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=-1,
                    basePosition=[1.5, 1.5, float(self.z_max)],
                    physicsClientId=CLIENT,
                )
                _track(ceiling)
            except Exception:
                pass

    def _preprocessAction(self, action):
        """Convert policy output to a PID target (with leash), and log diagnostics."""
        if self.ACT_TYPE == ActionType.PID:
            a = np.asarray(action, dtype=np.float32).reshape(self.NUM_DRONES, 3)
            raw_target = a[0]  # PPO’s absolute XYZ setpoint in meters
            
            pos  = self._getDroneStateVector(0)[0:3]
            dist = float(np.linalg.norm(self.goal - pos))
            
            base = np.array([0.05, 0.05, 0.04], dtype=np.float32)   # was 0.05
            boost = np.clip(0.10 * dist, 0.0, 0.10)                 # up to +0.10 m/step far away
            max_step_xyz = base + boost
            curr = self._getDroneStateVector(0)[0:3]
            # move at most max_step toward the raw target
            delta = np.clip(raw_target - pos, -max_step_xyz, +max_step_xyz)
            proposed = curr + delta

            if not hasattr(self, "_ema_target"):
                self._ema_target = curr.copy()
            alpha = 0.35
            self._ema_target = (1.0 - alpha) * self._ema_target + alpha * proposed
            smoothed = self._ema_target
            # ---- diagnostics for plotting (what NN asked vs what PID will chase)
            self._last_policy_action = raw_target.copy()
            self._last_pid_target    = smoothed.copy()

            # send to BaseAviary PID path
            out = np.zeros((self.NUM_DRONES, 3), dtype=np.float32)
            out[0] = smoothed
            return super()._preprocessAction(out)

        # other action types unchanged
        return super()._preprocessAction(action)
    
    def _clear_static(self):
        for bid in self._obstacle_ids:
            try: p.removeBody(bid, physicsClientId=self.CLIENT)
            except: pass
        self._obstacle_ids.clear()
        if self._goal_id is not None:
            try: p.removeBody(self._goal_id, physicsClientId=self.CLIENT)
            except: pass
        self._goal_id = None

    def _spawn_goal_marker(self):
        # small bright cube as goal marker (easy to see)
        self._goal_id = p.loadURDF("cube.urdf",
                                    self.goal.tolist(),
                                    p.getQuaternionFromEuler([0,0,0]),
                                    useFixedBase=True, globalScaling=0.15,
                                    physicsClientId=self.CLIENT)
        # paint it green
        try:
            p.changeVisualShape(self._goal_id, -1, rgbaColor=[0,1,0,1], physicsClientId=self.CLIENT)
        except:
            pass

    def _add_obstacles_for_difficulty(self):
    # remove previous
        for bid in getattr(self, "_obstacle_ids", []):
            try: p.removeBody(bid, physicsClientId=self.CLIENT)
            except: pass
        self._obstacle_ids = []

        def add_box(x,y,z, color=[0.8,0.2,0.2,1]):
            bid = p.loadURDF("cube.urdf", [x,y,z], useFixedBase=True, physicsClientId=self.CLIENT)
            try: p.changeVisualShape(bid, -1, rgbaColor=color, physicsClientId=self.CLIENT)
            except: pass
            self._obstacle_ids.append(bid)

        d = self.difficulty
        if d == 0:
        # open field = no obstacles
            return
        elif d == 1:
        # single wall (row of cubes)
            for i in range(5):
                add_box(x=1.0, y=0.5+0.3*i, z=0.25)
        elif d == 2:
        # corridor with a gap
            for i in range(6):
                add_box(x=1.0, y=-0.3+0.3*i, z=0.25)
                if i != 3:
                    add_box(x=1.8, y=-0.3+0.3*i, z=0.25)
        elif d == 3:
        # zig-zag obstacles
            for (x,y) in [(0.8,0.6),(1.4,1.0),(2.0,1.4)]:
                add_box(x,y,0.25,color=[0.9,0.4,0.1,1])
        else:
        # random clutter
            rng = np.random.default_rng()
            for _ in range(6):
                x = float(rng.uniform(0.6, 2.4))
                y = float(rng.uniform(0.4, 2.0))
                add_box(x,y,0.25,color=[0.7,0.7,0.2,1])