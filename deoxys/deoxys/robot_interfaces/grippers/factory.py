from .base import BaseGripper
from .xarm_gripper import XArmGripper
from .robotiq_gripper import RobotiqGripper


def make_gripper(gripper_type: str, robot_api=None) -> BaseGripper:
    """Gripper factory.

    Args:
        gripper_type: "xarm" or "robotiq"
        robot_api:    XArmAPI instance — required for "xarm", ignored for "robotiq".

    Returns:
        A connected BaseGripper instance.
    """
    if gripper_type == "xarm":
        if robot_api is None:
            raise ValueError("robot_api is required for XArmGripper.")
        return XArmGripper(robot_api)
    elif gripper_type == "robotiq":
        return RobotiqGripper()
    else:
        raise ValueError(
            f"Unknown gripper type: {gripper_type!r}. Choose 'xarm' or 'robotiq'."
        )

