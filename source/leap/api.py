import numpy as np

from .dynamixel.protocol import DynamixelDriverProtocol
from .utils import allegro_positions_to_leap

DEFAULT_JOINT_OFFSETS = -np.array(
    [
        3.1415927410125732,
        3.1415927410125732,
        4.71238899230957,
        4.71238899230957,
        6.2831854820251465,
        3.1415927410125732,
        3.1415927410125732,
        -0.0,
        3.1415927410125732,
        3.1415927410125732,
        3.1415927410125732,
        6.2831854820251465,
        4.71238899230957,
        3.1415927410125732,
        3.1415927410125732,
        3.1415927410125732,
    ]
)


class LeapHand:
    def __init__(
        self,
        driver: DynamixelDriverProtocol,
    ):
        self._driver = driver
        driver.set_operation_mode(5)
        driver.set_position_p_gain([450, 600, 600, 600, 450, 600, 600, 600, 450, 600, 600, 600, 450, 600, 600, 600])
        driver.set_position_d_gain([150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200])
        driver.set_goal_current([350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350])
        driver.set_torque_mode(enable=True)

    def set_joints_leap(self, joint_positions: np.ndarray):
        """
        Set the joint positions directly to control the Leap Hand without any conversion, only the offsets are applied.

        :param joint_positions: A numpy array of joint positions, expected to be of length 16.
        """
        if len(joint_positions) != 16:
            raise ValueError("Expected 16 joint positions.")

        self._driver.set_joint_positions(joint_positions)

    def set_positions_allegro(self, joint_positions: np.ndarray):
        """
        Set the joint positions for the Allegro Hand, applying the necessary offsets.

        :param joint_positions: A numpy array of joint positions, expected to be of length 16.
        """
        if len(joint_positions) != 16:
            raise ValueError("Expected 16 joint positions.")

        # Apply offsets to the joint positions
        self._driver.set_joint_positions(allegro_positions_to_leap(joint_positions))


if __name__ == "__main__":
    from .dynamixel.driver import DynamixelDriver

    driver = DynamixelDriver(
        servo_ids=list(range(16)),
        port="/dev/cu.usbserial-FTA2U4SR",
        baud_rate=4_000_000,
    )
    driver.set_torque_mode(enable=True)

    leap_hand = LeapHand(driver, joint_offsets=DEFAULT_JOINT_OFFSETS)
    leap_hand.set_joints_leap(np.zeros(16))
