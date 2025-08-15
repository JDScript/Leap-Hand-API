from collections.abc import Sequence
from typing import Protocol

import numpy as np


class DynamixelDriverProtocol(Protocol):
    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, *, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joint_positions(self) -> np.ndarray:
        """Get the current joint positions in radians.

        Returns:
            np.ndarray: An array of joint positions.
        """
        ...

    def set_joint_positions(self, joint_positions: Sequence[float]):
        """Set the joint positions for the Dynamixel servos.

        Args:
            joint_positions (Sequence[float]): A list of joint positions.
        """
        ...

    def get_joint_velocities(self) -> np.ndarray:
        """Get the current joint velocities in radians per second.

        Returns:
            np.ndarray: An array of joint velocities.
        """
        ...

    def get_joint_currents(self) -> np.ndarray:
        """Get the current joint currents in Amperes.

        Returns:
            np.ndarray: An array of joint currents.
        """
        ...

    def close(self):
        """Close the driver."""
