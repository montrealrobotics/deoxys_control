import time
import math
import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import zmq

logger = logging.getLogger(__name__)


def _finite_vec(x: List[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)


class XArmInterface:

    def __init__(
        self,
        robot_ip: str,
        cmd_port: int = 5555,
        dof: int = 6,
        control_freq: float = 30.0,
        control_timeout: float = 1.0,
        joint_map: Optional[List[int]] = None,
    ):
        self.dof = dof
        self._control_freq = control_freq
        self._control_interval = 1.0 / max(control_freq, 1e-6)
        self._control_timeout = control_timeout

        self.joint_map = joint_map if joint_map is not None else list(range(dof))
        if len(self.joint_map) != dof:
            raise ValueError("joint_map must have length DOF")

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{robot_ip}:{cmd_port}")

        time.sleep(0.3)

        self.last_time_ns: Optional[int] = None

    def preprocess(self):
        pass

    def _rate_limit_sleep(self, termination: bool):
        if self.last_time_ns is None:
            self.last_time_ns = time.time_ns()
            return
        if termination:
            return

        now_ns = time.time_ns()
        remaining = self._control_interval - (now_ns - self.last_time_ns) / 1e9
        if 0.0001 < remaining < self._control_timeout:
            time.sleep(remaining)
        self.last_time_ns = time.time_ns()

    def control(
        self,
        controller_type: str,
        action: Union[np.ndarray, list],
        controller_cfg: dict = None,
        termination: bool = False,
    ):
        """

        Supported controller_type:
          - "JOINT_POSITION"

        action:
          - joint angles in radians, must be same length as dof
        """
        self._rate_limit_sleep(termination)

        q = np.asarray(action, dtype=np.float64).reshape(-1)

        if q.size < self.dof:
            raise ValueError(f"Expected >= {self.dof} joint values, got {q.size}")

        q = q[: self.dof].tolist()

        if not _finite_vec(q):
            logger.warning("Ignoring command with NaN/Inf")
            return None

        q_xarm = [q[self.joint_map[i]] for i in range(self.dof)]

        payload = np.concatenate(([time.time()], np.array(q_xarm, dtype=np.float64))).astype(np.float64)

        self._pub.send(payload.tobytes())

        return None

    def close(self):
        try:
            self._pub.close(0)
        except Exception:
            pass
