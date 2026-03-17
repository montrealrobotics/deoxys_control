import threading
import time
import logging
from typing import Dict, Optional

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.state import RobotState
from utils.utils import Rate
from grippers import make_gripper, GRIPPER_OPEN

logger = logging.getLogger("xarm_control")


class XArmRobot:
    DEFAULT_MAX_DELTA = 0.05

    def __init__(
        self,
        ip: str = "192.168.42.222",
        real: bool = True,
        use_gripper: bool = True,
        dof: int = 6,
        control_frequency: float = 50.0,
        max_delta: float = DEFAULT_MAX_DELTA,
        gripper_type: str = "xarm",
    ):
        logger.info(f"Connecting to robot at {ip}")
        self.real = real
        self.use_gripper = use_gripper
        self.max_delta = max_delta
        self.dof = dof
        self.dof_arm = dof - 1 if use_gripper else dof
        self._control_frequency = control_frequency

        if real:
            from xarm.wrapper import XArmAPI
            from xarm.core import XCONF

            self.robot = XArmAPI(ip, is_radian=True)
            self.robot_config = XCONF()
            self.device_type = (
                int("{}1305".format(self.robot.axis))
                if self.robot.sn
                and 1305 <= int(self.robot.sn[2:6]) < 8500
                else self.robot.device_type
            )
            self.joint_limit = (
                self.robot_config.Robot.JOINT_LIMITS
                .get(self.robot.axis)
                .get(self.device_type, [])
            )
        else:
            self.robot = None

        self._clear_error_states()

        self._gripper_obj = None
        if use_gripper:
            self._gripper_obj = make_gripper(gripper_type, robot_api=self.robot)
            if real:
                self._gripper_obj.connect()
                self._gripper_obj.set_position(GRIPPER_OPEN, wait=True)

        self.last_state_lock = threading.Lock()
        self.target_command_lock = threading.Lock()
        self.last_state = self._update_last_state()
        self.target_command = {
            "joints": self.last_state.joints(),
            "gripper": GRIPPER_OPEN,
        }
        self.running = True
        self.command_thread = None
        if real:
            self.command_thread = threading.Thread(target=self.run, daemon=True)
            self.command_thread.start()

    def num_dofs(self) -> int:
        return self.dof

    def get_state(self) -> RobotState:
        with self.last_state_lock:
            return self.last_state

    def get_joint_state(self) -> np.ndarray:
        state = self.get_state()
        if self.use_gripper:
            return np.concatenate([state.joints(), [state.gripper_pos()]])
        return state.joints()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) == self.dof_arm:
            self.set_command(joint_state, None)
        elif len(joint_state) == self.dof:
            self.set_command(joint_state[: self.dof_arm], joint_state[self.dof_arm])
        else:
            raise ValueError(
                f"Invalid joint state length {len(joint_state)}: expected {self.dof_arm} or {self.dof}."
            )

    def set_command(self, joints: np.ndarray, gripper: Optional[float] = None) -> None:
        with self.target_command_lock:
            self.target_command = {"joints": joints, "gripper": gripper}

    def stop(self) -> None:
        self.running = False

    def get_observations(self) -> Dict[str, np.ndarray]:
        state = self.get_state()
        pos_quat = np.concatenate([state.cartesian_pos(), state.quat()])
        joints = self.get_joint_state()
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(state.gripper_pos()),
        }

    def run(self) -> None:
        rate = Rate(duration=1 / self._control_frequency)
        step_times = []
        count = 0

        while self.running:
            s_t = time.time()
            self.last_state = self._update_last_state()

            with self.target_command_lock:
                joint_delta = self.target_command["joints"] - self.last_state.joints()
                gripper_command = self.target_command["gripper"]

            norm = np.linalg.norm(joint_delta)
            delta = joint_delta / norm * self.max_delta if norm > self.max_delta else joint_delta

            if not np.all(delta == 0):
                self._set_position(self.last_state.joints() + delta)

            if self.use_gripper and gripper_command is not None:
                self._gripper_obj.set_position(gripper_command, wait=False)

            self.last_state = self._update_last_state()
            rate.sleep()

            step_times.append(time.time() - s_t)
            count += 1
            if count % 1000 == 0:
                freq = 1 / np.mean(step_times)
                logger.warning(
                    f"Control loop frequency — mean: {freq:10.3f} Hz"
                )
                step_times = []

    def _clear_error_states(self) -> None:
        if self.robot is None:
            return
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(1)
        self.robot.set_mode(1)
        time.sleep(1)
        self.robot.set_collision_sensitivity(0)
        time.sleep(1)
        self.robot.set_state(state=0)
        time.sleep(1)

    def _update_last_state(self) -> RobotState:
        with self.last_state_lock:
            if self.robot is None:
                return RobotState(
                    x=0.0, y=0.0, z=0.0, gripper=0.0,
                    joints_list=(0.0,) * self.dof,
                    aa=np.zeros(3),
                )

            gripper_pos = self._gripper_obj.get_position() if self.use_gripper else None

            code, servo_angle = self.robot.get_servo_angle(is_radian=True)
            servo_angle = servo_angle[: self.dof_arm]
            while code != 0:
                logger.error(f"get_servo_angle() error code {code}, retrying...")
                self._clear_error_states()
                code, servo_angle = self.robot.get_servo_angle(is_radian=True)
                servo_angle = servo_angle[: self.dof_arm]

            code, cart_pos = self.robot.get_position_aa(is_radian=True)
            while code != 0:
                logger.error(f"get_position_aa() error code {code}, retrying...")
                self._clear_error_states()
                code, cart_pos = self.robot.get_position_aa(is_radian=True)

            cart_pos = np.array(cart_pos)
            aa = cart_pos[3:]
            cart_pos[:3] /= 1000  # mm → m

            return RobotState.from_robot(cart_pos, servo_angle, gripper_pos, aa)

    def _set_position(self, joints: np.ndarray) -> None:
        if self.robot is None:
            return
        ret = self.robot.set_servo_angle_j(joints, wait=False, is_radian=True)
        if ret in [1, 9]:
            self._clear_error_states()

