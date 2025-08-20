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

    def set_operation_mode(self, mode: int):
        """Set the operation mode for the Dynamixel servos.

        Args:
            mode (int): The operation mode to set.
        """
        ...

    def set_position_p_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the P gain for the Dynamixel servos, with retries and better error logging.

        Args:
            gains (list[float]): A list of P gain values for each servo.
            retries (int): Number of retry attempts for each servo.
            retry_interval (float): Time interval between retries in seconds.
        """
        ...

    def set_velocity_p_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the velocity P gain for the Dynamixel servos, with retries and better error logging.

        Args:
            gains (list[float]): A list of velocity P gain values for each servo.
            retries (int): Number of retry attempts for each servo.
            retry_interval (float): Time interval between retries in seconds.
        """
        ...

    def set_position_d_gain(self, gains: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the position D gain for the Dynamixel servos, with retries and better error logging.

        Args:
            gains (list[float]): A list of position D gain values for each servo.
            retries (int): Number of retry attempts for each servo.
            retry_interval (float): Time interval between retries in seconds.
        """
        ...

    def set_goal_current(self, currents: list[float], retries: int = 3, retry_interval: float = 0.02):
        """Set the goal current for the Dynamixel servos, with retries and better error logging.

        Args:
            currents (list[float]): A list of goal current values for each servo.
            retries (int): Number of retry attempts for each servo.
            retry_interval (float): Time interval between retries in seconds.
        """
        ...

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: int | None = None,
        dxl_id: int | None = None,
        context: str | None = None,
    ) -> bool:
        """Handles the result from a communication request.

        Args:
            comm_result (int): The communication result code.
            dxl_error (int | None): The Dynamixel error code, if any.
            dxl_id (int | None): The servo ID associated with the error, if any.
            context (str | None): Additional context information for the error.

        Returns:
            bool: True if communication was successful, False otherwise.
        """
        ...

    def close(self):
        """Close the driver."""
