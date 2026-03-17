import time
import logging
from .base import BaseGripper

logger = logging.getLogger("xarm_control")


class XArmGripper(BaseGripper):
    """Built-in xArm gripper, communicates through the XArmAPI."""

    def __init__(self, robot_api):
        """
        Args:
            robot_api: An initialized XArmAPI instance.
        """
        self._robot = robot_api
        self.open_raw: float = 855
        self.closed_raw: float = -5


    def connect(self) -> None:
        self._robot.set_gripper_enable(True)
        time.sleep(1)
        self._robot.set_gripper_mode(0)
        time.sleep(1)
        self._robot.set_gripper_speed(3000)
        time.sleep(1)

        # Calibrate closed position
        self._robot.set_gripper_position(self.closed_raw, wait=True)
        _, self.closed_raw = self._robot.get_gripper_position()

        # Calibrate open position
        self._robot.set_gripper_position(self.open_raw, wait=True)
        _, self.open_raw = self._robot.get_gripper_position()

        logger.info(
            f"RobotiqGripper calibrated: open_raw={self.open_raw}, closed_raw={self.closed_raw}"
        )


    def get_position(self) -> float:
        code, raw = self._robot.get_gripper_position()
        while code != 0 or raw is None:
            logger.error(f"get_gripper_position() error code {code}, retrying...")
            time.sleep(0.001)
            code, raw = self._robot.get_gripper_position()
            if code == 22:
                raise RuntimeError("Gripper error 22 – clear robot errors first.")
        return self._to_normalized(raw)

    def set_position(self, pos: float, wait: bool = False) -> None:
        raw = self._to_raw(pos)
        self._robot.set_gripper_position(raw, wait=wait)

