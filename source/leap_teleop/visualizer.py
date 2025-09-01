"""
Visualization module for Leap Hand teleoperation using Rerun.
"""

from pathlib import Path

import numpy as np
import rerun as rr


class LeapHandVisualizer:
    def __init__(self, urdf_path: str):
        """Initialize the visualizer with URDF path."""
        self.urdf_path = Path(urdf_path)

        # Initialize rerun
        rr.init("leap_hand_teleop", spawn=True)

        # Log the robot model - we'll use rerun's built-in mesh support
        # For now we'll just visualize joint positions

    def log_joint_positions(self, joint_positions: np.ndarray, joint_names: list[str]):
        """Log current joint positions to rerun."""
        if len(joint_positions) != len(joint_names):
            raise ValueError(f"Number of joint positions ({len(joint_positions)}) must match number of joint names ({len(joint_names)})")

        # Log joint positions as text for now
        joint_text = f"Joints: {joint_positions[:4].round(2)}"  # Show first 4 joints
        rr.log("status/joints", rr.TextLog(joint_text))

    def log_hand_landmarks(self, keypoints_3d: np.ndarray):
        """Log detected hand landmarks to rerun."""
        if keypoints_3d is None:
            return

        # Log as 3D points
        rr.log("hand/landmarks", rr.Points3D(
            positions=keypoints_3d,
            colors=[255, 100, 100],
            radii=0.005
        ))

        # Define hand connections for visualization
        connections = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
        ]

        # Log connections as line segments
        for i, (start, end) in enumerate(connections):
            if start < len(keypoints_3d) and end < len(keypoints_3d):
                rr.log(f"hand/connections/{i}", rr.LineStrips3D(
                    strips=[[keypoints_3d[start], keypoints_3d[end]]],
                    colors=[100, 255, 100]
                ))

    def log_hand_detected_true(self):
        """Log that hand was detected."""
        rr.log("status/detection", rr.TextLog("Hand detected"))

    def log_hand_detected_false(self):
        """Log that hand was not detected."""
        rr.log("status/detection", rr.TextLog("No hand detected"))

    def log_status_success(self, message: str):
        """Log success status message."""
        rr.log("status/message", rr.TextLog(message))

    def log_status_error(self, message: str):
        """Log error status message."""
        rr.log("status/message", rr.TextLog(message))

    def log_leap_hand_state(self, joint_positions: np.ndarray, target_positions: np.ndarray = None):
        """Log Leap Hand state including current and target positions."""
        # Log current positions as text
        current_text = f"Current: {joint_positions[:4].round(2)}"
        rr.log("leap_hand/current", rr.TextLog(current_text))

        # Target positions if provided
        if target_positions is not None:
            target_text = f"Target: {target_positions[:4].round(2)}"
            rr.log("leap_hand/target", rr.TextLog(target_text))
