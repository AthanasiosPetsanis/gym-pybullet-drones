import numpy as np
from gymnasium import spaces
import pybullet as p
from collections import deque

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
        random_start: bool = False,
        start_center_xy=(0.0, 0.0),
        start_radius: float = 0.3,
        start_z_range=(0.75, 0.95),
        keep_goal_z_equal_spawn: bool = True,
        ceiling_margin: float = 0.6,
        pyb_freq: int = 240,
        ctrl_freq: int = 24,
    ):
        # difficulty / obstacles
        self.difficulty = int(difficulty)
        self._obstacles_enabled = bool(obstacles)
        self._obstacle_ids = []      # track bodies we spawn
        self._goal_id = None
        self._obstacle_specs = []    # record primitives for export

        # safety / ceiling
        self.z_max = 1.8           # hard ceiling height (m)
        self.use_ceiling = True    # turn the invisible ceiling on/off
        self.episode_sec = 25.0
        self.ceiling_margin = float(ceiling_margin)

        # success / HOLD
        self.success_thresh = 0.1     # meters
        self.hold_steps_req = 1       # must stay inside for this many steps to count success
        self._hold_count = 0

        # reward weights
        self.progress_w     = 20.0          # weight for progress (prev_dist - dist)
        self.dist_w         = 0.05          # constant pull toward the goal each step
        self.step_cost      = 0.01          # tiny time penalty per step
        self.success_bonus  = 1500.0         # one-time bonus for reaching the goal

        # spawn/goal controls
        self.random_start = bool(random_start)
        self.start_center_xy = tuple(start_center_xy)
        self.start_radius = float(start_radius)
        self.start_z_range = tuple(start_z_range)
        self.keep_goal_z_equal_spawn = bool(keep_goal_z_equal_spawn)

        # diagnostic / last action logging
        self._last_policy_action = None
        self._last_pid_target = None
        self._stuck_hist = deque(maxlen=48)   # ~2s ιστορικό σε 24 Hz
        self._last_pos_for_stuck = None
        self.hold_new_actions_until_close = False
        # ΜΕΙΩΝΟΥΜΕ το κατώφλι για να δέχεται πιο εύκολα νέο action (λιγότερο «κόλλημα» στο παλιό)
        self.reach_eps = 0.1
        self._held_action = None

        #OOB params
        self.oob_step_penalty   = 2.0    # μικρό per-step κόστος όταν είναι εκτός
        self.oob_term_penalty   = 300.0  # επιπλέον κόστος τη στιγμή του OOB τερματισμού

        # --- DEBUG SWITCHES ---
        # 1) Απενεργοποίηση stuck truncation για να μην κόβονται πρόωρα επεισόδια κατά το debug
        self.debug_no_stuck = False
        # 2) «Καθαρό» τεστ PID: αγνοεί την policy και πάει κατευθείαν στο goal
        self.force_pid_to_goal = False

        # --- RAW action → setpoint mapping (world-frame Δx,Δy,Δz) ---
        self.use_raw_action_mapping = True     # ενεργοποιεί το «καθαρό» mapping
        self.raw_step_scale = np.array([0.3, 0.3, 0.22], dtype=np.float32)  # m/step (x,y,z)

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
            # καθαρά οπτική παρατήρηση: 3 x H x W (uint8)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(3, self.IMG_RES[1], self.IMG_RES[0]),
                dtype=np.uint8,
            )

        if self.OBS_TYPE == ObservationType.KIN:
            base_space = self.observation_space           # ό,τι όρισε το BaseRLAviary
            base_dim = int(np.prod(base_space.shape))     # Flatten dimension
            self._base_obs_dim = base_dim                 # για debug αν χρειαστεί

            low = np.full((base_dim + 3,), -np.inf, dtype=np.float32)
            high = np.full((base_dim + 3,), np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ---- SAFE CLEANUP HELPERS ----
    def _p_connected(self) -> bool:
        try:
            return bool(p.isConnected(self.CLIENT))
        except Exception:
            return False

    def _safe_remove(self, bid):
        if bid is None or not self._p_connected():
            return
        try:
            p.getBodyInfo(bid, physicsClientId=self.CLIENT)  # raises if invalid
        except Exception:
            return
        try:
            p.removeBody(bid, physicsClientId=self.CLIENT)
        except Exception:
            pass

    def _safe_clear_episode_bodies(self):
        # clear obstacles
        if getattr(self, "_obstacle_ids", None):
            for bid in list(self._obstacle_ids):
                self._safe_remove(bid)
            self._obstacle_ids.clear()
        # clear goal
        if getattr(self, "_goal_id", None) is not None:
            self._safe_remove(self._goal_id)
            self._goal_id = None
        # clear specs
        if getattr(self, "_obstacle_specs", None) is not None:
            self._obstacle_specs.clear()
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
        goal_vec = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        goal_xyz = start + goal_vec
        if self.keep_goal_z_equal_spawn:
            goal_xyz[2] = start[2]

        self.goal = goal_xyz.astype(np.float32)
        self.INIT_XYZS = start.reshape(1, 3)

        # --- define a "room" bounding box to keep the agent in a corridor ---
        # margins [m] around the straight line from spawn->goal
        _x0 = float(start[0]); _y0 = float(start[1]); _z0 = float(start[2])
        _xg = float(self.goal[0]); _yg = float(self.goal[1]); _zg = float(self.goal[2])
        x_min = min(_x0, _xg) - 1.0
        x_max = max(_x0, _xg) + 3.0
        y_min = min(_y0, _yg) - 2.0
        y_max = max(_y0, _yg) + 2.0
        z_min = 0.45
        z_max = float(getattr(self, "z_max", 1.8))
        self._room_bbox = (x_min, x_max, y_min, y_max, z_min, z_max)

        # ceiling safety margin
        if getattr(self, "use_ceiling", True):
            self.z_max = float(max(self.z_max, start[2] + self.ceiling_margin))

        # (re)build geometry for this level
        self._build_level_geometry(start, self.goal)

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _computeObs(self):
        """
        Παρατήρηση:
        - KIN: [flattened base_obs, goal - pos]  → 1D vector
        - RGB: εικόνα κάμερας (3, H, W)
        """
        if self.OBS_TYPE == ObservationType.KIN:
            base_obs = super()._computeObs()
            base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)  # flatten

            pos = self._getDroneStateVector(0)[0:3]
            goal_vec = (self.goal - pos).astype(np.float32)                # (3,)

            obs = np.concatenate([base_obs, goal_vec], axis=0)
            return obs

        # RGB case
        img, _, _ = self._getDroneImages(0)
        return img

    def _computeReward(self):
        s   = self._getDroneStateVector(0)
        pos = s[0:3]

        dist = float(np.linalg.norm(self.goal - pos))
        if self._last_distance is None:
            progress = 0.0
        else:
            progress = self._last_distance - dist
        self._last_distance = dist

        reward  = 60.0 * progress            # δυνατό σήμα κατεύθυνσης
        reward += -0.04 * dist               # σταθερή έλξη
        reward += -0.003                     # μικρό step cost

        # σκαλοπάτια κοντά στο goal (πυκνώνουμε «καρότο» πριν το success)
        if dist < 1.5:  reward += 10.0
        if dist < 1.0:  reward += 35.0
        if dist < 0.8:  reward += 60.0
        if dist < 0.6:  reward += 90.0

        if dist < self.success_thresh:
            reward += 150.0                  # +20 σε σχέση με πριν, για καλύτερο σήμα

        if self._isCollision(0):
            reward -= 300.0

        # --- OOB penalties ---
        if self._is_oob(pos):
            reward -= self.oob_step_penalty
            # αν πρόκειται να τερματίσει τώρα ως OOB, ρίξε και επιπλέον terminal κόστος
            if getattr(self, "_term_reason", "") == "oob":
                reward -= self.oob_term_penalty

        vel = s[10:13]                         # γραμμική ταχύτητα από state
        gv  = (self.goal - pos)
        if np.linalg.norm(vel) > 1e-6 and np.linalg.norm(gv) > 1e-6:
            cos_align = float(np.dot(vel, gv) / (np.linalg.norm(vel)*np.linalg.norm(gv)))
        else:
            cos_align = 0.0
        reward += 0.5 * cos_align              # μικρό βάρος, ρυθμίσιμο

        # terminal success bonus (προσδοκώμενο hold στο ίδιο βήμα)
        prospective_hold = self._hold_count + 1 if dist < self.success_thresh else 0
        if prospective_hold >= self.hold_steps_req:
            reward += self.success_bonus

        return reward


    def _computeTerminated(self):
        pos  = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))

        # gates (level 3)
        all_gates_ok = True
        if getattr(self, "_gates", None) and self.difficulty == 3:
            all_gates_ok = (self._gates_passed >= len(self._gates))

        # hold-in-goal λογική
        if dist < self.success_thresh and all_gates_ok:
            self._hold_count = getattr(self, "_hold_count", 0) + 1
        else:
            self._hold_count = 0

        # επιτυχία
        if self._hold_count >= self.hold_steps_req:
            self._term_reason = "success"
            return True

        # OOB: αυστηρό κόψιμο όταν βγει εκτός «room»
        pos = self._getDroneStateVector(0)[0:3]
        if self._is_oob(pos):
            self._term_reason = "oob"
            return True

        # ΜΗΝ τερματίζεις για σύγκρουση τα πρώτα ~60 βήματα (≈2.5s @24Hz)
        collided = bool(self._isCollision(0))
        if self.step_counter > 60 and collided:
            self._term_reason = "collision"
            return True

        self._term_reason = ""
        return False


    def _computeTruncated(self):
        # Με το debug_no_stuck δεν κάνουμε stuck-truncation,
        # κρατάμε ΜΟΝΟ ένα μεγάλο time limit για ασφάλεια.
        if getattr(self, "debug_no_stuck", False):
            pyb = getattr(self, "pyb_freq", getattr(self, "PYB_FREQ", 240))
            ctrl = getattr(self, "ctrl_freq", getattr(self, "CTRL_FREQ", 24))
            time_limit = (self.step_counter >= 900 * (pyb / ctrl))  # ~9000 βήματα @24Hz
            return bool(time_limit)

        # (παλιό logic)
        pyb = getattr(self, "pyb_freq", getattr(self, "PYB_FREQ", 240))
        ctrl = getattr(self, "ctrl_freq", getattr(self, "CTRL_FREQ", 24))

        # OOB hard truncation (keeps eval stable and avoids 35m "escapes")
        pos = self._getDroneStateVector(0)[0:3]
        if self._is_oob(pos):
            self._term_reason = "oob"
            return True
        # stuck detection
        pos  = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))
        self._stuck_hist.append(dist)
        stuck = (len(self._stuck_hist) == self._stuck_hist.maxlen and
                (np.max(self._stuck_hist) - np.min(self._stuck_hist) < 0.08))

        time_limit = (self.step_counter >= 900 * (pyb / ctrl))
        return bool(stuck or time_limit)


    def _computeInfo(self):
        pos  = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))

        # Level-3 gate pass-through
        if getattr(self, "_gates", None) and self.difficulty == 3 and self._gates_passed < len(self._gates):
            x_plane, y_center, z_hole, R_hole = self._gates[self._gates_passed]
            near_plane = abs(float(pos[0] - x_plane)) <= 0.18
            dy, dz = float(pos[1]-y_center), float(pos[2]-z_hole)
            in_circle = (dy*dy + dz*dz) <= (R_hole - 0.05)**2
            if near_plane and in_circle:
                self._gates_passed += 1

        info = {
            "distance_to_goal": dist,
            "is_success": bool(self._hold_count >= self.hold_steps_req),
            "gates_passed": int(getattr(self, "_gates_passed", 0)),
            "num_gates": int(len(getattr(self, "_gates", []))),
            "pos": pos.copy(),
            "hold_count": int(getattr(self, "_hold_count", 0)),
            "collision": bool(self._isCollision(0)),
            "term_reason": getattr(self, "_term_reason", ""),
        }
        if getattr(self, "_last_pid_target", None) is not None:
            info["pid_target"] = self._last_pid_target.copy()
        if getattr(self, "_last_policy_action", None) is not None:
            info["policy_action"] = self._last_policy_action.copy()
        if getattr(self, "_held_action", None) is not None:
            info["held_action"] = self._held_action.copy()
        # Lightweight console debug κάθε ~1s @24Hz
        try:
            if (self.step_counter % 24) == 0:
                _held  = getattr(self, "_held_action", None)
                _sp    = getattr(self, "_last_pid_target", None)
                print(f"[t={self.step_counter:04d}] dist={dist:.2f}  pos={pos}  "
                    f"held={_held if _held is not None else 'None'}  "
                    f"sp={_sp if _sp is not None else 'None'}  "
                    f"term={getattr(self,'_term_reason','')}")
        except Exception:
            pass    
        return info


    # ---------- Back-compat shim ----------
    def _build_level_geometry(self, start, goal):
        self.goal = np.array(goal, dtype=np.float32)
        self._addObstacles()

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
    
    def _is_oob(self, pos):
        if not hasattr(self, "_room_bbox"):
            return False
        x_min, x_max, y_min, y_max, z_min, z_max = self._room_bbox
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        return (x < x_min) or (x > x_max) or (y < y_min) or (y > y_max) or (z < z_min) or (z > z_max)


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
        self._safe_clear_episode_bodies()

        # ----- ensure bookkeeping
        if not hasattr(self, "_obstacle_ids"):
            self._obstacle_ids = []
        if not hasattr(self, "_goal_id"):
            self._goal_id = None
        if not hasattr(self, "_obstacle_specs"):
            self._obstacle_specs = []

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

        def _record_box(center, half, rgba):
            self._obstacle_specs.append({"type":"box", "center":list(center), "half":list(half), "rgba":list(rgba)})

        def _record_cylinder(center, radius, height, rgba):
            self._obstacle_specs.append({"type":"cylinder", "center":list(center), "radius":float(radius), "height":float(height), "rgba":list(rgba)})

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
            _record_cylinder([x, y, float(height/2.0)], radius, height, rgba)
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
            _record_box(center_xyz, half_extents, color_rgba)
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

            # half sizes for the 4 pieces
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
                _record_box([1.5, 1.5, float(self.z_max)], [5,5,0.02], [0.7,0.7,0.7,0.0])
            except Exception:
                pass

    def _preprocessAction(self, action):
        if self.ACT_TYPE != ActionType.PID:
            return super()._preprocessAction(action)

        # --- RAW mapping: PPO action -> Δx,Δy,Δz (world frame) ---
        a = np.asarray(action, dtype=np.float32).reshape(self.NUM_DRONES, 3)[0]
        a = np.clip(a, -1.0, 1.0)
        self._last_policy_action = a.copy()  # log

        pos = self._getDroneStateVector(0)[0:3]

        # κλιμάκωση του ωμού action σε μέτρα ανά control step
        scale = getattr(self, "raw_step_scale", np.array([0.45, 0.45, 0.30], dtype=np.float32))
        step = scale * a

        # clamp βήματος για ασφάλεια (προαιρετικό σφίξιμο)
        step[0] = np.clip(step[0], -0.60, 0.60)
        step[1] = np.clip(step[1], -0.60, 0.60)
        step[2] = np.clip(step[2], -0.30, 0.30)

        sp_raw = pos + step

        # όριο ύψους (ασφάλεια)
        z_lo = 0.5
        z_hi = float(getattr(self, "z_max", 1.8)) - 0.10
        sp_raw[2] = np.clip(sp_raw[2], z_lo, z_hi)

        # προαιρετικό «hold μέχρι κοντά» (αν είναι ενεργό στο __init__)
        if getattr(self, "hold_new_actions_until_close", False):
            if getattr(self, "_held_action", None) is None:
                self._held_action = sp_raw.copy()
            else:
                if float(np.linalg.norm(pos - self._held_action)) >= float(self.reach_eps):
                    sp_raw = self._held_action.copy()
                else:
                    self._held_action = sp_raw.copy()
        else:
            self._held_action = sp_raw.copy()

        sp = sp_raw.astype(np.float32)
        self._last_pid_target = sp.copy()

        # το BaseRLAviary περιμένει στόχο ανά drone (NUM_DRONES,3)
        out = np.zeros((self.NUM_DRONES, 3), dtype=np.float32)
        out[0] = sp
        return super()._preprocessAction(out)

    def export_current_level_to_obj(self, out_path: str, cyl_segments: int = 24):
        """
        Export all recorded primitives (boxes/cylinders) into one Wavefront OBJ.
        Works with Blender's built-in 'Wavefront (.obj)' importer.
        """
        specs = getattr(self, "_obstacle_specs", [])
        if not specs:
            print("[export] nothing to export (build a level first)"); return

        verts = []   # (x,y,z)
        faces = []   # (i1,i2,i3) 1-based indices for OBJ
        colors = []  # per-face RGB (we'll bake approximate color; Blender will import as a single object)

        def add_box(center, half, rgba):
            cx, cy, cz = center; hx, hy, hz = half
            # 8 corners
            v = [
                (cx-hx, cy-hy, cz-hz),(cx+hx, cy-hy, cz-hz),(cx+hx, cy+hy, cz-hz),(cx-hx, cy+hy, cz-hz),
                (cx-hx, cy-hy, cz+hz),(cx+hx, cy-hy, cz+hz),(cx+hx, cy+hy, cz+hz),(cx-hx, cy+hy, cz+hz),
            ]
            base = len(verts)
            verts.extend(v)
            # 12 triangles (two per face)
            quads = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
            for q in quads:
                faces.append((base+q[0]+1, base+q[1]+1, base+q[2]+1))
                faces.append((base+q[0]+1, base+q[2]+1, base+q[3]+1))
                colors.append(tuple(rgba[:3])); colors.append(tuple(rgba[:3]))

        def add_cylinder(center, radius, height, rgba, nseg=cyl_segments):
            cx, cy, cz = center
            r = radius; h = height
            base = len(verts)
            # ring vertices (cylinder axis along X)
            for k in range(nseg):
                th = 2*np.pi*k/nseg
                x = cx
                y = cy + r*np.cos(th)
                z = cz + r*np.sin(th)
                verts.append((x, y, z))
            for k in range(nseg):
                th = 2*np.pi*k/nseg
                x = cx + h
                y = cy + r*np.cos(th)
                z = cz + r*np.sin(th)
                verts.append((x, y, z))
            # side faces
            for k in range(nseg):
                a  = base + k + 1
                b  = base + ((k+1)%nseg) + 1
                a2 = base + nseg + k + 1
                b2 = base + nseg + ((k+1)%nseg) + 1
                faces.append((a, b, b2)); colors.append(tuple(rgba[:3]))
                faces.append((a, b2, a2)); colors.append(tuple(rgba[:3]))

        # collect from recorded specs
        for s in specs:
            if s["type"] == "box":
                add_box(s["center"], s["half"], s.get("rgba",[0.7,0.7,0.7,1]))
            elif s["type"] == "cylinder":
                add_cylinder(s["center"], s["radius"], s["height"], s.get("rgba",[0.85,0.35,0.35,1]))
            else:
                continue

        # write OBJ (single object)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# vision track export\n")
            for vx,vy,vz in verts:
                f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            # OBJ has no per-face color; we ignore colors here—set material in Blender
            for a,b,c in faces:
                f.write(f"f {a} {b} {c}\n")
        print(f"[export] OBJ saved: {out_path}  (verts={len(verts)}, faces={len(faces)})")