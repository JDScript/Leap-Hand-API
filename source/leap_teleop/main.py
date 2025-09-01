"""
Leap Hand Teleoperation using hand pose detection and retargeting.
"""

from pathlib import Path
import time

import cv2
from dex_retargeting.constants import HandType
from dex_retargeting.constants import RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from loguru import logger
import numpy as np
import tyro

from source.leap.api import LeapHand
from source.leap.dynamixel.driver import DynamixelDriver

from .constants import mapping
from .hand_detector import SingleHandDetector
from .visualizer import LeapHandVisualizer


class LeapHandTeleop:
    def __init__(
        self,
        retargeting_type: RetargetingType = RetargetingType.vector,
        hand_type: HandType = HandType.right,
        *,
        robot_urdf_path: str | None = None,
        enable_real_robot: bool = False,
        servo_port: str = "/dev/cu.usbserial-FTA2U4SR",
        baud_rate: int = 4_000_000,
    ):
        """Initialize Leap Hand teleoperation system."""
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.enable_real_robot = enable_real_robot

        # Set up paths
        if robot_urdf_path is None:
            robot_urdf_path = str(Path(__file__).parent / "assets" / "leap_right" / "leap_hand_right.urdf")
        self.robot_urdf_path = robot_urdf_path

        # Initialize hand detector
        hand_type_str = "Right" if hand_type == HandType.right else "Left"
        self.detector = SingleHandDetector(hand_type=hand_type_str, selfie=False)

        # Initialize retargeting system
        self._setup_retargeting()

        # Initialize visualizer
        self.visualizer = LeapHandVisualizer(self.robot_urdf_path)

        # Initialize real robot if requested
        self.leap_hand = None
        if enable_real_robot:
            self._setup_real_robot(servo_port, baud_rate)

        logger.info(f"Leap Hand Teleoperation initialized with {hand_type_str.lower()} hand")

    def _setup_retargeting(self):
        """Set up the retargeting system."""
        # Create a custom config since we have our own URDF
        config = None
        if self.retargeting_type == RetargetingType.vector:
            config = RetargetingConfig(
                type=self.retargeting_type.name,
                urdf_path=self.robot_urdf_path,
                target_origin_link_names=["base", "base", "base", "base"],
                target_task_link_names=["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"],
                wrist_link_name="base",
                scaling_factor=1.6,
                target_link_human_indices=np.array([[0, 0, 0, 0], [4, 8, 12, 16]]),
                low_pass_alpha=0.1,
            )
        if self.retargeting_type == RetargetingType.dexpilot:
            config = RetargetingConfig(
                type=self.retargeting_type.name,
                urdf_path=self.robot_urdf_path,
                # target_origin_link_names=["base", "base", "base", "base"],
                # target_task_link_names=["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"],
                wrist_link_name="base",
                finger_tip_link_names=["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"],
                scaling_factor=1.6,
                # target_link_human_indices=np.array([[0, 0, 0, 0], [4, 8, 12, 16]]),
                low_pass_alpha=0.1,
            )

        # Build retargeting system directly from our config
        self.retargeting = config.build()
        self.joint_names = self.retargeting.joint_names
        logger.info(f"Retargeting system initialized with {len(self.joint_names)} joints")

    def _setup_real_robot(self, servo_port: str, baud_rate: int):
        """Set up connection to real Leap Hand."""
        try:
            servo_ids = list(range(16))
            driver = DynamixelDriver(
                servo_ids=servo_ids,
                port=servo_port,
                baud_rate=baud_rate,
                reading_interval=1,
            )
            self.leap_hand = LeapHand(driver)
            logger.success(f"Connected to Leap Hand on {servo_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Leap Hand: {e}")
            self.leap_hand = None

    def run_camera_teleop(self, camera_id: int = 0, fps: int = 30):
        """Run teleoperation using camera input."""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info("Starting camera teleoperation. Press 'q' to quit, 'space' to toggle robot control")

        robot_control_enabled = False
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue

                frame_count += 1

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect hand
                num_hands, joint_pos, keypoint_2d, _ = self.detector.detect(rgb_frame)

                if joint_pos is not None:
                    # Retarget to robot
                    retargeting_type = self.retargeting.optimizer.retargeting_type
                    indices = self.retargeting.optimizer.target_link_human_indices

                    if retargeting_type == "POSITION":
                        ref_value = joint_pos[indices, :]
                    else:
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                    # Get robot joint positions
                    robot_qpos = self.retargeting.retarget(ref_value)

                    # Send to robot if enabled and connected
                    if robot_control_enabled and self.leap_hand is not None:
                        # Convert from allegro format to leap format
                        self.leap_hand.set_positions_allegro(robot_qpos[mapping])

                    # Visualize
                    self.visualizer.log_hand_landmarks(joint_pos)
                    self.visualizer.log_hand_detected_true()
                    self.visualizer.log_joint_positions(robot_qpos, self.joint_names)

                    if self.leap_hand is not None:
                        # current_positions = self.leap_hand._driver.get_joint_positions()
                        # self.visualizer.log_leap_hand_state(current_positions, leap_positions)
                        pass

                    status = f"Hand detected, robot {'ON' if robot_control_enabled else 'OFF'}"
                    self.visualizer.log_status_success(status)

                    # Draw skeleton on image
                    frame = self.detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")
                else:
                    self.visualizer.log_hand_detected_false()
                    self.visualizer.log_status_error("No hand detected")

                # Add status text to frame
                status_text = f"Robot: {'ON' if robot_control_enabled else 'OFF'} | Press SPACE to toggle"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Leap Hand Teleoperation", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):  # Space bar
                    robot_control_enabled = not robot_control_enabled
                    logger.info(f"Robot control {'enabled' if robot_control_enabled else 'disabled'}")

                time.sleep(1.0 / fps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.leap_hand is not None:
                # Move to safe position
                safe_position = np.zeros(16)
                self.leap_hand.set_positions_allegro(safe_position)
                logger.info("Moved robot to safe position")


def main(
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    robot_urdf_path: str | None = "vendor/dex-urdf/robots/hands/leap_hand/leap_hand_right.urdf",
    *,
    enable_real_robot: bool = False,
    servo_port: str = "/dev/cu.usbserial-FTA2U4SR",
    baud_rate: int = 4_000_000,
    camera_id: int = 0,
    fps: int = 30,
):
    """
    Leap Hand Teleoperation using hand pose detection.

    Args:
        retargeting_type: Type of retargeting algorithm to use
        hand_type: Which hand to track (left or right)
        robot_urdf_path: Path to robot URDF file (optional, uses default if not provided)
        enable_real_robot: Whether to connect to and control real robot
        servo_port: Serial port for robot connection
        baud_rate: Baud rate for serial communication
        camera_id: Camera device ID
        fps: Frames per second for camera and control loop
    """
    teleop = LeapHandTeleop(
        retargeting_type=retargeting_type,
        hand_type=hand_type,
        robot_urdf_path=robot_urdf_path,
        enable_real_robot=enable_real_robot,
        servo_port=servo_port,
        baud_rate=baud_rate,
    )

    teleop.run_camera_teleop(camera_id=camera_id, fps=fps)


if __name__ == "__main__":
    tyro.cli(main)
