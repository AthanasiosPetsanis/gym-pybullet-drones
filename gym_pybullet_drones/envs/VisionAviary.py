import numpy as np
from gymnasium import spaces
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class VisionAviary(BaseRLAviary):
    def __init__(self, gui=False, record=False, act=None, obs=None, obstacles=True,
                 initial_xyzs=None, initial_rpys=None,
                 pyb_freq: int = 240, ctrl_freq: int = 30):
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

        self._obstacles_enabled = obstacles
        if self._obstacles_enabled:
            self._addObstacles()
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.control_timestep = 1/60    
        self.IMG_RES = (256, 256)
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL
        self.current_action = None
        self.action_timer = 0
        self.goal = np.array([2.0, 4.0, 1.0])
        #self.action_space=act
        #self.observation_space=obs
        #self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        #self.observation_space = spaces.Box(
            #low=0.0,
            #high=255.0,
            #shape=(3 * self.IMG_RES[0] * self.IMG_RES[1] + 3,),
            #dtype=np.float32
        #)

    def _getDroneImages(self, drone_id=0, segmentation=False):
        pos = self._getDroneStateVector(drone_id)[:3]
        drone_orientation = self._getDroneStateVector(drone_id)[3:6]

        # Set the camera slightly behind and above the drone, facing forward
        offset = np.array([0, 0, 0.2])
        camera_pos = pos + offset
        target_pos = pos + self._getDroneForwardVector(drone_id)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.IMG_RES[0]) / self.IMG_RES[1],
            nearVal=0.1,
            farVal=10.0
        )
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=self.RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        rgbImg = np.reshape(rgbImg, (height, width, 4))[:, :, :3]
        rgbImg = np.transpose(rgbImg, (2, 0, 1))
        return rgbImg, depthImg, segImg

    def _getDroneForwardVector(self, drone_id=0):
        orientation = self._getDroneStateVector(drone_id)[3:7]  # quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        forward_vector = rot_matrix @ np.array([1, 0, 0])
        return forward_vector

    def _computeObs(self):
        rgb, _, _ = self._getDroneImages(0)
        drone_pos = self._getDroneStateVector(0)[0:3]
        rel_goal = self.goal - drone_pos
        rgb = rgb.astype(np.float32).flatten() / 255.0  # Normalize image
        obs = np.concatenate([rgb, rel_goal.astype(np.float32)])
        return obs

    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(self.goal - drone_pos)
        reward = -1.0  # step penalty
        reward -= 2.0 * distance
        if distance < 0.3:
            reward += 100.0
        if self._isCollision(0):
            reward -= 100.0
        return reward

    def _computeTerminated(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(self.goal - drone_pos)
        if distance < 0.3:
            return True
        if self._isCollision(0):
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter >= 300

    def _computeInfo(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        return {"distance_to_goal": float(np.linalg.norm(self.goal - drone_pos))}

    def _preprocessAction(self, action):
    # Από [0, 1] => [0, 20000] RPM
        rpms = np.clip(action, 0.0, 1.0) * 20000

    # Debug
        #print(f"[Step {self.step_counter}] Raw action: {action}")
        #print(f"[Step {self.step_counter}] Scaled RPMs: {np.round(rpms, 1)}")

        return rpms.reshape(1, 4)

    def _isCollision(self, drone_id=0):
        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)
        return len(contacts) > 0

    def _addObstacles(self):
        obstacle_positions = [
            [2, 2, 0.25],
            [3, 3, 0.25],
            [1, 1, 0.25],
            [3, 1, 0.25],
            [1, 3, 0.25]
        ]
        for pos in obstacle_positions:
            p.loadURDF(
                "cube.urdf",
                pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                globalScaling=1.0,
                physicsClientId=self.CLIENT
            )