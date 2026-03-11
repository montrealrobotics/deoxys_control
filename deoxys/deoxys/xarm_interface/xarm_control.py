import dataclasses
import threading
import time
from typing import Dict, Optional, Tuple
import sys
import logging
import numpy as np
from pyquaternion import Quaternion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xarm_control")

class Transformations:
    @staticmethod
    def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
        """Convert a quaternion to an axis-angle representation.

        Args:
            quat (np.ndarray): The quaternion to convert.

        Returns:
            np.ndarray: The axis-angle representation of the quaternion.
        """
        assert quat.shape == (4,), "Input quaternion must be a 4D vector."
        norm = np.linalg.norm(quat)
        assert norm != 0, "Input quaternion must not be a zero vector."
        quat = quat / norm

        Q = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
        angle = Q.angle
        axis = Q.axis
        aa = axis * angle
        return aa


    def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
        """Convert an axis-angle representation to a quaternion.

        Args:
            aa (np.ndarray): The axis-angle representation to convert.

        Returns:
            np.ndarray: The quaternion representation of the axis-angle.
        """
        assert aa.shape == (3,), "Input axis-angle must be a 3D vector."

        norm = np.linalg.norm(aa)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0])

        axis = aa / norm

        Q = Quaternion(axis=axis, angle=norm)
        quat = np.array([Q.x, Q.y, Q.z, Q.w])
        return quat


@dataclasses.dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    z: float
    gripper: float
    joints_list: Tuple[float, ...]
    aa: np.ndarray

    @staticmethod
    def from_robot(
        cartesian: np.ndarray,
        joints_list: np.ndarray,
        gripper: float,
        aa: np.ndarray,
    ) -> "RobotState":
        return RobotState(
            x=cartesian[0],
            y=cartesian[1],
            z=cartesian[2],
            gripper=gripper,
            joints_list=np.array(joints_list),
            aa=aa,
        )

    def cartesian_pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def quat(self) -> np.ndarray:
        return Transformations._quat_from_aa(self.aa)

    def joints(self) -> np.ndarray:
        return np.array(self.joints_list)

    def gripper_pos(self) -> float:
        return self.gripper


class Rate:
    def __init__(self, *, duration):
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


class XArmRobot(object):
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 100
    DEFAULT_MAX_DELTA = 0.05
    def __init__(
        self,
        ip: str = "192.168.42.222",
        real: bool = True,
        use_gripper: bool = True,
        dof: int = 6,
        control_frequency: float = 50.0,
        max_delta: float = DEFAULT_MAX_DELTA,
        use_robotiq: bool = False,
    ):
        logger.info(ip)
        self.real = real
        self.use_gripper = use_gripper
        self.use_robotiq = use_robotiq
        self.max_delta = max_delta
        self.dof = dof
        if self.use_gripper:
            self.dof_arm = dof - 1
        else:
            self.dof_arm = dof

        if real:
            from xarm.wrapper import XArmAPI
            from xarm.core import XCONF

            self.robot = XArmAPI(ip, is_radian=True)
            self.robot_config = XCONF()

            self.device_type = int('{}1305'.format(self.robot.axis)) if self.robot.sn and int(self.robot.sn[2:6]) >= 1305 and int(self.robot.sn[2:6]) < 8500 else self.robot.device_type
            self.joint_limit = self.robot_config.Robot.JOINT_LIMITS.get(self.robot.axis).get(self.device_type, [])
        else:
            self.robot = None

        if self.use_robotiq:
            import pyRobotiqGripper
            gripper = pyRobotiqGripper.RobotiqGripper()
            self.gripper = gripper
            gripper.activate()

        self._control_frequency = control_frequency
        self._clear_error_states()
        if self.use_gripper:
            self._set_gripper_position(self.GRIPPER_OPEN)

        self.last_state_lock = threading.Lock()
        self.target_command_lock = threading.Lock()
        self.last_state = self._update_last_state()
        self.target_command = {
            "joints": self.last_state.joints(),
            "gripper": 0,
        }
        self.running = True
        self.command_thread = None
        if real:
            self.command_thread = threading.Thread(target=self.run)
            self.command_thread.start()

    def num_dofs(self) -> int:
        return self.dof

    def get_joint_state(self) -> np.ndarray:
        state = self.get_state()
        if self.use_gripper:
            gripper = state.gripper_pos()
            all_dofs = np.concatenate([state.joints(), np.array([gripper])])
        else:
            all_dofs = state.joints()
        return all_dofs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) == self.dof_arm:
            self.set_command(joint_state, None)
        elif len(joint_state) == self.dof:
            self.set_command(joint_state[:self.dof_arm], joint_state[self.dof])
        else:
            raise ValueError(
                f"Invalid joint state: {joint_state}, len={len(joint_state)}"
            )

    def stop(self):
        self.running = False


    def get_state(self) -> RobotState:
        with self.last_state_lock:
            return self.last_state

    def set_command(self, joints: np.ndarray, gripper: Optional[float] = None) -> None:
        with self.target_command_lock:
            self.target_command = {
                "joints": joints,
                "gripper": gripper,
            }

    def _clear_error_states(self):
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
        if self.use_gripper:
            if self.use_robotiq:
                pass
            else:
                self.robot.set_gripper_enable(True)
                time.sleep(1)
                self.robot.set_gripper_mode(0)
                time.sleep(1)
                self.robot.set_gripper_speed(3000)
                time.sleep(1)

    def _get_gripper_pos(self) -> float:
        if self.robot is None:
            return 0.0
        if self.use_robotiq:
            return self.gripper.getPosition() / 255
        else:
            code, gripper_pos = self.robot.get_gripper_position()
            while code != 0 or gripper_pos is None:
                logger.error(f"Error code {code} in get_gripper_position(). {gripper_pos}")
                time.sleep(0.001)
                code, gripper_pos = self.robot.get_gripper_position()
                if code == 22:
                    self._clear_error_states()

            normalized_gripper_pos = (gripper_pos - self.GRIPPER_OPEN) / (
                self.GRIPPER_CLOSE - self.GRIPPER_OPEN
            )
            return normalized_gripper_pos

    def _set_gripper_position(self, pos: int) -> None:
        if self.robot is None:
            return
        if self.use_robotiq:
            pos = 255 - (pos / 800 * 255)
            try:
                self.gripper.goTo(pos, wait=False)
            except Exception as e:
                logger(e)
                logger.error(pos)
                raise e
        else:
            self.robot.set_gripper_position(pos, wait=False)

    def run(self):
        rate = Rate(
            duration=1 / self._control_frequency
        )  # command and update rate for robot
        step_times = []
        count = 0

        while self.running:
            s_t = time.time()
            # update last state
            self.last_state = self._update_last_state()
            with self.target_command_lock:
                joint_delta = np.array(
                    self.target_command["joints"] - self.last_state.joints()
                )
                gripper_command = self.target_command["gripper"]

            norm = np.linalg.norm(joint_delta)
            # threshold delta to be at most 0.01 in norm space
            if norm > self.max_delta:
                delta = joint_delta / norm * self.max_delta
            else:
                delta = joint_delta
            if not all(d == 0 for d in delta):
                target_joints = self.last_state.joints() + delta

                # command position
                self._set_position(
                    target_joints
                )
            if self.use_gripper:
                if gripper_command is not None:
                    set_point = gripper_command
                    self._set_gripper_position(
                        self.GRIPPER_OPEN
                        + set_point * (self.GRIPPER_CLOSE - self.GRIPPER_OPEN)
                    )
            self.last_state = self._update_last_state()

            rate.sleep()
            step_times.append(time.time() - s_t)
            count += 1
            if count % 1000 == 0:
                # Mean, Std, Min, Max, only show 3 decimal places and string pad with 10 spaces
                frequency = 1 / np.mean(step_times)
                logger.warn(
                    f"Low  Level Frequency - mean: {frequency:10.3f}, std: {np.std(frequency):10.3f}, min: {np.min(frequency):10.3f}, max: {np.max(frequency):10.3f}"
                )
                step_times = []

    def _update_last_state(self) -> RobotState:
        with self.last_state_lock:
            if self.robot is None:
                return RobotState(x=0.0,
                                  y=0.0,
                                  z=0.0,
                                  gripper=0.0,
                                  joints_list=(0.0,) * self.dof,
                                  aa=np.zeros(3)
                                  )
            if self.use_gripper:
                gripper_pos = self._get_gripper_pos()
            else:
                gripper_pos = None

            code, servo_angle = self.robot.get_servo_angle(is_radian=True)
            servo_angle = servo_angle[: self.dof_arm]
            while code != 0:
                logger.error(f"Error code {code} in get_servo_angle().")
                self._clear_error_states()
                code, servo_angle = self.robot.get_servo_angle(is_radian=True)
                servo_angle = servo_angle[: self.dof_arm-1]

            code, cart_pos = self.robot.get_position_aa(is_radian=True)
            while code != 0:
                logger(f"Error code {code} in get_position().")
                self._clear_error_states()
                code, cart_pos = self.robot.get_position_aa(is_radian=True)

            cart_pos = np.array(cart_pos)
            aa = cart_pos[3:]
            cart_pos[:3] /= 1000

            return RobotState.from_robot(
                cart_pos,
                servo_angle,
                gripper_pos,
                aa,
            )

    def _set_position(
        self,
        joints: np.ndarray,
    ) -> None:
        if self.robot is None:
            return
        ret = self.robot.set_servo_angle_j(joints, wait=False, is_radian=True)
        if ret in [1, 9]:
            self._clear_error_states()

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


if __name__ == "__main__":
    # if len(sys.argv) < 1:
    #     print('Usage: {} {{robot_ip}} {{use_gripper}}'.format(sys.argv[0]))
    #     print('  use_gripper: true/false')
    #     exit(1)

    # robot_ip = sys.argv[1]
    # use_gripper = sys.argv[2]
    robot_ip = "192.168.42.222"
    use_gripper = False
    logger.info('**********************************************************************')
    logger.info('* robot_ip: {}'.format(robot_ip))
    logger.info('**********************************************************************')
    teleop = XArmRobot(robot_ip, use_gripper=use_gripper)
    import time

    time.sleep(1)
    logger.info(teleop.get_state())
    logger.info(teleop.get_state())
    logger.info(teleop.get_state())
    logger.info(teleop.get_state())
    logger.info(teleop.get_state())

    time.sleep(1)
    state = teleop.get_state()
    current_joints = state.joints()
    logger.info(current_joints)
    joint_command = current_joints.copy()
    for i in range(30):
        joint_command[0] -= 0.01
        logger.info(f'current joints: {current_joints}')
        logger.info(f'command I am sending: {joint_command}')
        teleop.set_command(joint_command)
        state = teleop.get_state()

        time.sleep(0.05)
    for i in range(30):
        joint_command[0] += 0.01
        logger.info(f'current joints: {current_joints}')
        logger.info(f'command I am sending: {joint_command}')
        teleop.set_command(joint_command)
        state = teleop.get_state()

        time.sleep(0.05)
    new_joints = state.joints()
    logger.info(new_joints)
