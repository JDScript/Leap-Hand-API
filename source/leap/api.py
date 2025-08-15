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
        joint_offsets: np.ndarray = None,
    ):
        self._driver = driver
        self._joint_offsets = joint_offsets if joint_offsets is not None else np.zeros(16)
        assert len(self._joint_offsets) == 16, "Expected 16 joint offsets."

    def set_joints_leap(self, joint_positions: np.ndarray):
        """
        Set the joint positions directly to control the Leap Hand without any conversion, only the offsets are applied.

        :param joint_positions: A numpy array of joint positions, expected to be of length 16.
        """
        if len(joint_positions) != 16:
            raise ValueError("Expected 16 joint positions.")

        # Apply offsets to the joint positions
        adjusted_positions = joint_positions - self._joint_offsets
        self._driver.set_joint_positions(adjusted_positions)

    def set_positions_allegro(self, joint_positions: np.ndarray):
        """
        Set the joint positions for the Allegro Hand, applying the necessary offsets.

        :param joint_positions: A numpy array of joint positions, expected to be of length 16.
        """
        if len(joint_positions) != 16:
            raise ValueError("Expected 16 joint positions.")

        # Apply offsets to the joint positions
        adjusted_positions = joint_positions - self._joint_offsets
        self._driver.set_joint_positions(allegro_positions_to_leap(adjusted_positions))

    def calibrate_offsets(self):
        current_positions = self._driver.get_joint_positions()
        if len(current_positions) != 16:
            raise ValueError("Expected 16 joint positions.")

        offsets = current_positions - np.pi
        offsets = np.round(offsets / (np.pi / 2)) * (np.pi / 2)

        self._joint_offsets = offsets
        print(f"Calibrated joint offsets: {self._joint_offsets}")


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
