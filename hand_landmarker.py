import time

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from source.leap.dynamixel.driver import DynamixelDriver


class MediaPipeLeapHandMapping:
    def __init__(self):
        self.setup_dynamixel_driver()
        self.setup_mediapipe()
        print("MediaPipe to Leap Hand initialized")

    def setup_dynamixel_driver(self):
        servo_ids = list(range(16))
        self.driver = DynamixelDriver(
            servo_ids=servo_ids,
            port="/dev/cu.usbserial-FTA2U4SR",
            baud_rate=4_000_000,
            reading_interval=1,
        )
        self.driver.set_operation_mode(5)
        self.driver.set_p_gain([450, 600, 600, 600, 450, 600, 600, 600, 450, 600, 600, 600, 450, 600, 600, 600])
        self.driver.set_position_d_gain(
            [150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200, 150, 200, 200, 200]
        )
        self.driver.set_goal_current([350] * 16)

    def setup_mediapipe(self):
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if not detection_result.hand_landmarks:
            return rgb_image

        annotated_image = np.copy(rgb_image)
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
            )

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            height, width, _ = annotated_image.shape
            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]
            text_x = int(min(xs) * width)
            text_y = int(min(ys) * height) - 10

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name} Hand",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (88, 205, 54),
                2,
            )
        return annotated_image

    def ik(self, landmarks: np.ndarray):
        return np.zeros(16)  # Placeholder for actual IK computation

    def run(self):
        cap = cv2.VideoCapture(0)
        print("MediaPipe to Leap Hand running")
        print("Press 'q' to quit, 'r' to reset reference pose, 'c' to calibrate current pose as zero")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(image)

                if detection_result.hand_landmarks:
                    if not self.driver.torque_enabled():
                        self.driver.set_torque_mode(enable=True)

                    hand_landmarks = detection_result.hand_landmarks[0]
                    joint_angles = self.ik(hand_landmarks)

                    try:
                        self.driver.set_joint_positions(joint_angles + np.pi)
                    except Exception as e:
                        print(f"Failed to send joint positions: {e}")

                annotated_image = self.draw_landmarks_on_image(frame, detection_result)
                cv2.imshow("MediaPipe to Leap Hand", annotated_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    try:
                        self._ema_state = None
                        # self.driver.set_joint_positions(self.zero_offset + np.pi)
                        print("Reference pose reset (send command skipped)")
                    except Exception as e:
                        print(f"Failed to reset reference pose: {e}")

                time.sleep(0.03)

        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            print("Shutting down...")
            # with contextlib.suppress(Exception):
            #     self.driver.set_torque_mode(enable=False)
            cap.release()
            cv2.destroyAllWindows()
            print("System closed safely")


def main():
    mapper = MediaPipeLeapHandMapping()
    mapper.run()


if __name__ == "__main__":
    main()
