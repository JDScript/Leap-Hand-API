import os
import time

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pybullet as p

from source.leap.dynamixel.driver import DynamixelDriver


class MediaPipeLeapHandIK:
    def __init__(self, *, is_left=False, use_gui=False):
        # 初始化 Dynamixel 驱动器
        self.setup_dynamixel_driver()

        # 初始化 PyBullet
        self.setup_pybullet(use_gui)

        # 初始化 Leap Hand 模型
        self.setup_leap_hand(is_left)

        # 初始化 MediaPipe
        self.setup_mediapipe()

        # 设置参数
        self.is_left = is_left
        self.scaling_factor = 0.8  # 缩放因子，调整人手到机械手的尺寸映射

        # 平滑滤波参数
        self.alpha = 0.3
        self.prev_joint_positions = None

        print(f"Initialized, using {'left' if is_left else 'right'} hand mode")

    def setup_dynamixel_driver(self):
        """Initialize Dynamixel driver"""
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
        self.driver.set_goal_current([350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350])

    def setup_pybullet(self, use_gui):
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

    def setup_leap_hand(self, is_left):
        """加载 Leap Hand URDF 模型"""
        # 获取当前文件目录
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)

        if is_left:
            urdf_path = os.path.join(path_src, "leap_hand_mesh_left/robot_pybullet.urdf")
            position = [0.0, 0.0, 0.0]
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        else:
            urdf_path = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
            position = [0.0, 0.0, 0.0]
            orientation = p.getQuaternionFromEuler([0, 0, 0])

        # 如果没有URDF文件，创建一个简化的手部模型用于测试
        if not os.path.exists(urdf_path):
            print(f"警告：URDF文件 {urdf_path} 不存在，使用简化模型")
            self.leap_id = self.create_simple_hand_model()
        else:
            self.leap_id = p.loadURDF(urdf_path, position, orientation, useFixedBase=True)

        self.num_joints = p.getNumJoints(self.leap_id)

        # Leap Hand 末端执行器索引 (指尖位置)
        # 根据 Leap Hand 的运动学结构调整
        self.end_effector_indices = [3, 7, 11, 15, 19]  # 拇指、食指、中指、无名指、小指的指尖

        print(f"加载 Leap Hand 模型，关节数：{self.num_joints}")

    def create_simple_hand_model(self):
        """创建简化的手部模型用于测试"""
        # 这里可以创建一个简化的手部模型
        # 实际使用时请确保有正确的 URDF 文件
        return

    def setup_mediapipe(self):
        """初始化 MediaPipe 手部检测"""
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_fingertip_positions(self, hand_landmarks):
        """从 MediaPipe 手部关键点提取指尖位置"""
        # MediaPipe 手部关键点索引
        FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指指尖
        WRIST_INDEX = 0

        wrist = hand_landmarks[WRIST_INDEX]
        fingertips = []

        for tip_idx in FINGERTIP_INDICES:
            tip = hand_landmarks[tip_idx]
            # 相对于手腕的位置，并应用缩放
            relative_pos = [
                (tip.x - wrist.x) * self.scaling_factor,
                (tip.y - wrist.y) * self.scaling_factor,
                (tip.z - wrist.z) * self.scaling_factor,
            ]
            fingertips.append(relative_pos)

        return fingertips

    def extract_finger_joint_positions(self, hand_landmarks):
        """提取手指关节位置用于更精确的IK"""
        # MediaPipe 手部关键点索引
        FINGER_JOINTS = {
            "thumb": [1, 2, 3, 4],  # 拇指
            "index": [5, 6, 7, 8],  # 食指
            "middle": [9, 10, 11, 12],  # 中指
            "ring": [13, 14, 15, 16],  # 无名指
            "pinky": [17, 18, 19, 20],  # 小指
        }

        WRIST_INDEX = 0
        wrist = hand_landmarks[WRIST_INDEX]

        target_positions = []

        for finger_name, joint_indices in FINGER_JOINTS.items():
            # 只取指尖和中间关节作为目标点
            tip_idx = joint_indices[-1]  # 指尖
            mid_idx = joint_indices[-2]  # 指尖前一个关节

            # 指尖位置
            tip = hand_landmarks[tip_idx]
            tip_pos = [
                (tip.x - wrist.x) * self.scaling_factor,
                (tip.y - wrist.y) * self.scaling_factor,
                (tip.z - wrist.z) * self.scaling_factor,
            ]

            # 中间关节位置
            mid = hand_landmarks[mid_idx]
            mid_pos = [
                (mid.x - wrist.x) * self.scaling_factor,
                (mid.y - wrist.y) * self.scaling_factor,
                (mid.z - wrist.z) * self.scaling_factor,
            ]

            target_positions.extend([mid_pos, tip_pos])

        return target_positions

    def compute_ik(self, target_positions):
        """使用 PyBullet 计算逆运动学"""
        if self.leap_id is None:
            return [0.0] * 16

        p.stepSimulation()

        # 使用 PyBullet 的逆运动学求解器
        try:
            joint_poses = p.calculateInverseKinematics2(
                self.leap_id,
                self.end_effector_indices,
                target_positions[: len(self.end_effector_indices)],
                solver=p.IK_DLS,
                maxNumIterations=50,
                residualThreshold=0.0001,
            )

            # 将结果映射到16个关节
            leap_joint_positions = self.map_to_leap_joints(joint_poses)

            return leap_joint_positions

        except Exception as e:
            print(f"IK 计算失败: {e}")
            return [0.0] * 16

    def map_to_leap_joints(self, joint_poses):
        """将 PyBullet IK 结果映射到 Leap Hand 的16个关节"""
        # 根据 Leap Hand 的具体关节结构进行映射
        # 这个映射需要根据你的 URDF 文件和实际的关节顺序调整

        real_robot_hand_q = np.array([0.0] * 16)

        if len(joint_poses) >= 16:
            # 直接映射前16个关节
            real_robot_hand_q = np.array(joint_poses[:16])
        else:
            # 如果关节数不足，需要进行插值或重映射
            for i in range(min(len(joint_poses), 16)):
                real_robot_hand_q[i] = joint_poses[i]

        # 根据参考代码进行关节重映射
        # 这些重映射是为了适应 Leap Hand 的特殊运动学结构
        if len(real_robot_hand_q) >= 16:
            # 每个手指的前两个关节顺序调整
            real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]  # 拇指
            real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]  # 食指
            real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]  # 中指
            real_robot_hand_q[12:14] = real_robot_hand_q[12:14][::-1]  # 小指

        return [float(i) for i in real_robot_hand_q]

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """在图像上绘制手部关键点"""
        if not detection_result.hand_landmarks:
            return rgb_image

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # 绘制手部关键点
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # 添加手部标签
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 10

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (88, 205, 54),
                1,
                cv2.LINE_AA,
            )

        return annotated_image

    def run_teleoperation(self):
        """运行主循环"""
        cap = cv2.VideoCapture(0)

        print("开始手部遥操作（基于 PyBullet IK），按 'q' 退出")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为RGB并检测手部
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(image)

                # 处理检测到的手部
                if detection_result.hand_landmarks:
                    if not self.driver.torque_enabled():
                        self.driver.set_torque_mode(enable=True)

                    # 获取第一只手的关键点
                    hand_landmarks = detection_result.hand_landmarks[0]

                    # 提取目标位置
                    target_positions = self.extract_fingertip_positions(hand_landmarks)

                    # 计算逆运动学
                    joint_positions = self.compute_ik(target_positions)

                    # 应用平滑滤波
                    if self.prev_joint_positions is not None:
                        joint_positions = [
                            self.alpha * new + (1 - self.alpha) * old
                            for new, old in zip(joint_positions, self.prev_joint_positions, strict=False)
                        ]
                    self.prev_joint_positions = joint_positions.copy()

                    print(f"关节角度: {[f'{angle:.3f}' for angle in joint_positions]}")

                    # 发送到机械手
                    try:
                        # 根据需要调整偏移量
                        offset_positions = [pos + np.pi for pos in joint_positions]
                        self.driver.set_joint_positions(offset_positions)
                    except Exception as e:
                        print(f"发送关节角度失败: {e}")

                # 显示标注后的图像
                annotated_image = self.draw_landmarks_on_image(frame, detection_result)
                cv2.imshow("MediaPipe to Leap Hand IK", annotated_image)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.03)  # 控制帧率

        except KeyboardInterrupt:
            print("接收到中断信号")

        finally:
            # 清理资源
            print("关闭遥操作系统...")
            self.driver.set_torque_mode(enable=False)
            cap.release()
            cv2.destroyAllWindows()
            p.disconnect()
            print("系统已安全关闭")


def main():
    # 创建遥操作系统实例
    # use_gui=True 可以显示 PyBullet 可视化界面
    ik_system = MediaPipeLeapHandIK(is_left=False, use_gui=False)
    ik_system.run_teleoperation()


if __name__ == "__main__":
    main()
