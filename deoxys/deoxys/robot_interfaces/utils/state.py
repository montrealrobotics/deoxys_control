import dataclasses
from typing import Tuple
import numpy as np
from pyquaternion import Quaternion


class Transformations:
    @staticmethod
    def _aa_from_quat(quat: np.ndarray) -> np.ndarray:
        """Convert a quaternion to an axis-angle representation."""
        assert quat.shape == (4,), "Input quaternion must be a 4D vector."
        norm = np.linalg.norm(quat)
        assert norm != 0, "Input quaternion must not be a zero vector."
        quat = quat / norm

        Q = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
        return Q.axis * Q.angle

    @staticmethod
    def _quat_from_aa(aa: np.ndarray) -> np.ndarray:
        """Convert an axis-angle representation to a quaternion."""
        assert aa.shape == (3,), "Input axis-angle must be a 3D vector."
        norm = np.linalg.norm(aa)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0])
        Q = Quaternion(axis=aa / norm, angle=norm)
        return np.array([Q.x, Q.y, Q.z, Q.w])


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
            joints_list=tuple(joints_list),
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

