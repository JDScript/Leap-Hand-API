from collections.abc import Sequence

import numpy as np

from .protocol import DynamixelDriverProtocol


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_positions = np.zeros(len(ids), dtype=float)
        self._joint_velocities = np.zeros(len(ids), dtype=float)
        self._joint_currents = np.zeros(len(ids), dtype=float)
        self._torque_enabled = False

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, *, enable: bool):
        self._torque_enabled = enable

    def get_joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    def set_joint_positions(self, joint_positions: Sequence[float]):
        if len(joint_positions) != len(self._ids):
            raise ValueError("The length of joint_positions must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint positions")
        self._joint_positions = np.array(joint_positions)

    def get_joint_velocities(self) -> np.ndarray:
        return self._joint_velocities.copy()

    def get_joint_currents(self) -> np.ndarray:
        return self._joint_currents.copy()

    def close(self):
        pass
