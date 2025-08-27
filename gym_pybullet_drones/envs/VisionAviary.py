import numpy as np
from gymnasium import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class VisionAviary(BaseRLAviary):
    def __init__(self, gui=False, record=False, act=None, obs=None, obstacles=True,
                 initial_xyzs=None, initial_rpys=None, pyb_freq: int = 240, ctrl_freq: int = 24):
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=initial_xyzs,
            physics=Physics.PYB,
            gui=gui,
            record=record,
            act=act,          # <- expect ActionType('pid') so action_space = Box(3,)
            obs=obs,          # label only; we return an image below
            initial_rpys=initial_rpys,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
        )

        # PID controllers: one per drone (BaseRLAviary indexes self.ctrl[k])
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.NUM_DRONES)]

        # Safer spawn height if none provided
        if initial_xyzs is None:
            self.INIT_XYZS = np.array([[0.0, 0.0, 1.2]], dtype=np.float32)

        # Camera config
        self.IMG_RES = (84, 84)  # (W, H)
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL

        # Goal and simple obstacles (optional)
        self.goal = np.array([2.0, 4.0, 1.0], dtype=np.float32)
        self._obstacles_enabled = obstacles
        if self._obstacles_enabled:
            self._addObstacles()

        # Progress-reward state
        self._last_distance = None
        self._last_action = np.zeros(3, dtype=np.float32)

        # === Observation space: channel-first image (uint8) ===
        # SB3 CnnPolicy expects images; NatureCNN normalizes internally.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8
        )
        # Do NOT override action_space: BaseRLAviary sets it from 'act' (pid => Box(3,))

    # ---------- RL API ----------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        pos = self._getDroneStateVector(0)[0:3]
        self._last_distance = float(np.linalg.norm(self.goal - pos))
        self._last_action[:] = 0.0
        return obs, info

    def _computeObs(self):
        rgb, _, _ = self._getDroneImages(0)  # (3,H,W) uint8
        return rgb

    def _computeReward(self):
        pos = self._getDroneStateVector(0)[0:3]
        dist = float(np.linalg.norm(self.goal - pos))
        progress = 0.0 if self._last_distance is None else (self._last_distance - dist)

        reward = 5.0 * progress                    # move toward the goal
        if dist < 0.30:  reward += 100.0           # success bonus
        if pos[2] < 0.50: reward -= 20.0           # low altitude penalty
        if self._isCollision(0): reward -= 100.0   # collision penalty

        self._last_distance = dist
        return reward

    def _computeTerminated(self):
        pos = self._getDroneStateVector(0)[0:3]
        if np.linalg.norm(self.goal - pos) < 0.30:
            return True
        if self._isCollision(0):
            return True
        return False

    def _computeTruncated(self):
        # ~12.5 s at 24 Hz control
        return self.step_counter >= 300

    def _computeInfo(self):
        pos = self._getDroneStateVector(0)[0:3]
        return {"distance_to_goal": float(np.linalg.norm(self.goal - pos))}

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
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]           # HWC uint8
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)   # -> CHW uint8
        return rgb, None, None

    def _isCollision(self, drone_id=0):
        cps = p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)
        return len(cps) > 0

    def _addObstacles(self):
        obstacle_positions = [
            [2, 2, 0.25],
            [3, 3, 0.25],
            [1, 1, 0.25],
            [3, 1, 0.25],
            [1, 3, 0.25],
        ]
        for pos in obstacle_positions:
            p.loadURDF(
                "cube.urdf", pos, p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True, globalScaling=1.0, physicsClientId=self.CLIENT
            )