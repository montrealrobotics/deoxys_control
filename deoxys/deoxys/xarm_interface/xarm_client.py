#!/usr/bin/env python3
import math
import time
import logging
from typing import List, Optional

import numpy as np
import zmq

from xarm_controller import XArmRobot, Rate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XArmBridge:
    def __init__(self, robot_ip="192.168.42.222", dof=6):
        self.dof = dof
        self.zmq_sub_addr = "tcp://0.0.0.0:5555"
        self.zmq_pub_addr = "tcp://0.0.0.0:5556"
        self.timeout_s = 0.35
        self.control_rate = 100

        self.robot = XArmRobot(
            ip=robot_ip,
            real=True,
            use_gripper=False,
            dof=dof,
            control_frequency=50.0
        )

        ctx = zmq.Context.instance()

        # Subscriber for Commands
        self.sub_sock = ctx.socket(zmq.SUB)
        self.sub_sock.bind(self.zmq_sub_addr)
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_sock.setsockopt(zmq.RCVHWM, 1)

        # Publisher for State
        self.pub_sock = ctx.socket(zmq.PUB)
        self.pub_sock.bind(self.zmq_pub_addr)
        self.pub_sock.setsockopt(zmq.SNDHWM, 1)

        logger.info(f"Bridge Active: SUB {self.zmq_sub_addr} | PUB {self.zmq_pub_addr}")

        self._latest_q: Optional[np.ndarray] = None
        self._latest_rx_mono: float = 0.0

    def run(self):
        rate = Rate(
            duration=1 / self.control_rate
        )
        try:
            while True:
                loop_start = time.time()
                try:
                    while True:
                        msg = self.sub_sock.recv(zmq.NOBLOCK)
                        arr = np.frombuffer(msg, dtype=np.float64)
                        if arr.size == (1 + self.dof):
                            self._latest_q = arr[1:]
                            self._latest_rx_mono = time.monotonic()
                except zmq.Again:
                    pass

                if self._latest_q is not None:
                    if (time.monotonic() - self._latest_rx_mono) <= self.timeout_s:
                        self.robot.set_command(self._latest_q)

                obs = self.robot.get_observations()

                state_packet = np.concatenate([
                    np.array([time.time()]),
                    obs["joint_positions"],
                    obs["ee_pos_quat"]
                ]).astype(np.float64)
                print(state_packet)
                self.pub_sock.send(state_packet.tobytes())

                rate.sleep()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.robot.stop()

if __name__ == "__main__":
    bridge = XArmBridge(robot_ip="192.168.42.222", dof=6)
    bridge.run()
