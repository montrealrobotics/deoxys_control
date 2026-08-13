import time
import math
import logging
import threading
from typing import Optional, List, Union, Dict
from deoxys.robot_interfaces.utils.utils import ActionType

import numpy as np
import zmq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xarm_interface")

def _finite_vec(x: List[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in x)


class XArmInterface:
    def __init__(
        self,
        general_cfg: str,
        control_freq: float = 30.0,
        state_freq: float = 100.0,
        control_timeout: float = 1.0,
        dof: int = 6,
        has_gripper: bool = False,
        use_visualizer: bool = False,
        automatic_gripper_reset: bool=False,
        joint_map: Optional[List[int]] = None,
    ):
        self.dof_arm = dof
        self.dof = dof
        self._ctrl_ip = general_cfg.CTRL_HOST.IP_ETH
        self._cmd_port = general_cfg.CTRL_HOST.ARM_SUB_PORT
        self._state_port = general_cfg.CTRL_HOST.ARM_PUB_PORT
        self._control_freq = control_freq
        self._state_freq = state_freq
        self._control_interval = 1.0 / max(control_freq, 1e-6)
        self._control_timeout = control_timeout
        if has_gripper:
            self.dof += 1
        self.joint_map = joint_map if joint_map is not None else list(range(self.dof))
        self.has_gripper = has_gripper

        self.use_visualizer = use_visualizer
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(f"tcp://{self._ctrl_ip}:{self._cmd_port}")

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{self._ctrl_ip}:{self._state_port}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.RCVHWM, 1)

        self._latest_state = None
        self._state_lock = threading.Lock()
        self._running = True

        self._state_thread = threading.Thread(target=self._receive_state, daemon=True)
        self._state_thread.start()

        time.sleep(0.3)
        self.last_time = None

    def _receive_state(self):
        """Background thread to keep the state buffer clean and current."""
        while self._running:
            try:
                # Use a while loop to clear the buffer and get only the absolute newest packet
                msg = None
                while True:
                    try:
                        msg = self._sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    if msg:
                        data = np.frombuffer(msg, dtype=np.float64)
                        if data.size >= (1 + self.dof + 7):
                            with self._state_lock:
                                self._latest_state = {
                                    "timestamp": data[0],
                                    "joint_positions": data[1:1+self.dof_arm],
                                    "gripper_pos": data[1+self.dof_arm:2+self.dof_arm],
                                    "ee_pos": data[2+self.dof_arm:2+self.dof_arm+3],
                                    "ee_quat": data[2+self.dof_arm+3:2+self.dof_arm+7]
                            }

            except Exception as e:
                logger.error(f"Error receiving state: {e}")
            time.sleep(0.001)

    def last_state(self) -> Optional[Dict]:
        """Returns the most recent robot state."""
        with self._state_lock:
            return self._latest_state

    def control(
        self,
        controller_type: str,
        action: Union[np.ndarray, list],
        action_type = ActionType.delta,
        controller_cfg: dict = None,
        termination: bool = False,
    ):
        self._rate_limit_sleep(termination)

        q = np.asarray(action, dtype=np.float64).reshape(-1)
        if q.size < self.dof:
            raise ValueError(f"Expected >= {self.dof} joint values, got {q.size}")

        q = q[: self.dof].tolist()
        if not _finite_vec(q):
            logger.warning("Ignoring command with NaN/Inf")
            return None

        # Re-map joints if necessary
        q_xarm = [q[self.joint_map[i]] for i in range(self.dof)]

        payload = np.concatenate(([time.time()], np.array(q_xarm, dtype=np.float64), [action_type])).astype(np.float64)
        self._pub.send(payload.tobytes())
        return None

    def _rate_limit_sleep(self, termination: bool):
        if self.last_time is None:
            self.last_time = time.time_ns()
            return
        if termination:
            return
        now_ns = time.time_ns()
        remaining = self._control_interval - (now_ns - self.last_time) / 1e9
        if 0.0001 < remaining:
            time.sleep(remaining)
        self.last_time = time.time_ns()

    def close(self):
        self._running = False
        try:
            self._pub.close(0)
            self._sub.close(0)
        except Exception:
            pass
