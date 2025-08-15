import numpy as np


def allegro_positions_to_leap(joint_positions: np.ndarray) -> np.ndarray:
    """
    Convert joint positions from Allegro Hand format to Leap Hand format.
    :param joint_positions: A numpy array of joint positions in Allegro format.
    :return: A numpy array of joint positions in Leap format.
    """
    if len(joint_positions) != 16:
        raise ValueError("Expected 16 joint positions.")

    return joint_positions + np.pi

def leap_positions_to_allegro(joint_positions: np.ndarray) -> np.ndarray:
    """
    Convert joint positions from Leap Hand format to Allegro Hand format.
    :param joint_positions: A numpy array of joint positions in Leap format.
    :return: A numpy array of joint positions in Allegro format.
    """
    if len(joint_positions) != 16:
        raise ValueError("Expected 16 joint positions.")

    return joint_positions - np.pi
