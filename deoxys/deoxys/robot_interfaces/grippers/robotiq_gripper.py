import time
import logging
from .base import BaseGripper
from pyrobotiqgripper import RobotiqGripper as _RobotiqGripper

logger = logging.getLogger("robotiq_control")


class RobotiqGripper(BaseGripper):
    """Robotiq gripper via pyrobotiqgripper.

    """

    def __init__(self):
        self._gripper = _RobotiqGripper()
        self.open_raw: float = 0
        self.closed_raw: float = 255

    def connect(self) -> None:
        self._gripper.connect()
        self._gripper.resetActivate()
        time.sleep(5)

        # Calibrate closed position
        self._gripper.move(self.closed_raw, wait=True)
        self.closed_raw = self._gripper.getPosition()

        # Calibrate open position
        self._gripper.move(self.open_raw, wait=True)
        self.open_raw = self._gripper.getPosition()

        logger.info(
            f"RobotiqGripper calibrated: open_raw={self.open_raw}, closed_raw={self.closed_raw}"
        )

    def get_position(self) -> float:
        raw = self._gripper.getPosition()
        return self._to_normalized(raw)

    def set_position(self, pos: float, wait: bool = False) -> None:
        raw = int(self._to_raw(pos))
        try:
            self._gripper.move(raw, wait=wait)
        except Exception as e:
            logger.error(f"RobotiqGripper.set_position failed: {e} (raw={raw})")
            raise

