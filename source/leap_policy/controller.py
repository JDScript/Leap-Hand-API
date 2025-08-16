from pathlib import Path
import time

import numpy as np
import torch

from source.leap.api import LeapHand
from source.leap.dynamixel.driver import DynamixelDriver
from source.leap_policy.policy import RLGamesPolicy


class LeapHandController(LeapHand):
    def __init__(self, driver, control_freq: int = 10):
        super().__init__(driver)
        self.policy: RLGamesPolicy | None = None
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq

        self.device = "mps"

        self.init_pose = self.fetch_grasp_state()
        self.action_scale = 1 / 5
        self.action_type = "relative"
        self.leap_dof_lower, self.leap_dof_upper = self.get_leap_hand_joint_limits()
        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)

        self.hist_len = 3

    def restore_policy(self, policy: RLGamesPolicy):
        self.policy = policy
        self.policy.reset_hidden_state()

    def leap_sim_to_leap_hand(self, joints):
        return joints + 3.14159

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = torch.tensor(
            [4, 0, 8, 12, 6, 2, 10, 14, 7, 3, 11, 15, 1, 5, 9, 13], device=self.device
        )
        self.real_to_sim_indices = torch.tensor(
            [1, 12, 5, 9, 0, 13, 4, 8, 2, 14, 6, 10, 3, 15, 7, 11], device=self.device
        )

    def get_leap_hand_joint_limits(self):
        upper_limits = [
            2.2300,
            2.0940,
            2.2300,
            2.2300,
            1.0470,
            2.4430,
            1.0470,
            1.0470,
            1.8850,
            1.9000,
            1.8850,
            1.8850,
            2.0420,
            1.8800,
            2.0420,
            2.0420,
        ]
        lower_limits = [
            -0.3140,
            -0.3490,
            -0.3140,
            -0.3140,
            -1.0470,
            -0.4700,
            -1.0470,
            -1.0470,
            -0.5060,
            -1.2000,
            -0.5060,
            -0.5060,
            -0.3660,
            -1.3400,
            -0.3660,
            -0.3660,
        ]
        return lower_limits, upper_limits

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.real_to_sim_indices]
        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.sim_to_real_indices]
        return values[:, self.sim_to_real_indices]

    def leap_hand_to_sim_ones(self, joints):
        joints = self.leap_hand_to_leap_sim(joints)
        sim_min, sim_max = self.leap_sim_limits()
        return self.unscale_np(joints, sim_min, sim_max)

    def leap_hand_to_leap_sim(self, joints):
        return joints - 3.14159

    def leap_sim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        return sim_min, sim_max

    def unscale_np(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def command_joint_position(self, desired_pose):
        desired_pose = self.leap_sim_to_leap_hand(desired_pose)
        desired_pose = self.sim_to_real(desired_pose)
        desired_pose = desired_pose.detach().cpu().numpy().astype(float).flatten()

        self.set_joints_leap(desired_pose)

    def poll_joint_position(self):
        # read position from hardware
        joint_position = self._driver.get_joint_positions()
        joint_position = torch.from_numpy(joint_position.astype(np.float32)).to(device=self.device)

        joint_position = self.leap_hand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower

        return {"position": joint_position}

    def forward_network(self, obs):
        return self.policy.step(obs)["selected_action"]

    def run(self):
        assert self.policy is not None, "Policy must be restored before running"

        print("Command to the initial position")
        for _ in range(self.control_freq * 4):
            self.command_joint_position(self.init_pose)
            robot_state = self.poll_joint_position()
            obses = robot_state["position"]
            time.sleep(self.control_dt)
        print("Initial position reached!")

        # Get current state
        self.command_joint_position(self.init_pose)
        robot_state = self.poll_joint_position()
        obses = robot_state["position"]

        obs_hist_buf = torch.zeros((1, 32, self.hist_len), device=self.device, dtype=torch.float32)
        prev_target = obses.clone()

        unscaled_pos = self.unscale_np(obses, self.leap_dof_lower, self.leap_dof_upper)
        frame = torch.cat([unscaled_pos, prev_target], dim=-1).float()

        # Fill history buffer
        for i in range(self.hist_len):
            obs_hist_buf[0, :, i] = frame
        obs_hist_buf[0, :, -1] = frame
        obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float()

        counter = 0
        print("Starting policy execution:")
        try:
            while True:
                counter += 1
                start_time = time.time()

                # Get action from policy
                action = self.forward_network(obs_buf)
                action = action.squeeze(0)

                if self.action_type == "relative":
                    action = torch.clamp(action, -1.0, 1.0)
                    target = prev_target + self.action_scale * action
                elif self.action_type == "absolute":
                    action = self.unscale_np(action, self.leap_dof_lower, self.leap_dof_upper)
                    target = self.action_scale * action + (1.0 - self.action_scale) * prev_target
                else:
                    raise ValueError(f"Unsupported action type: {self.action_type}. Must be relative or absolute.")

                target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
                prev_target = target.clone()

                print(f"Sending command: {target}")
                self.command_joint_position(target)
                # self.command_joint_position(self.init_pose)

                robot_state = self.poll_joint_position()
                print(f"Received state: {robot_state['position']}")

                obses = robot_state["position"]
                unscaled_pos = self.unscale_np(obses, self.leap_dof_lower, self.leap_dof_upper)

                frame = torch.cat([unscaled_pos, target], dim=-1).float()
                obs_hist_buf[:, :, :-1] = obs_hist_buf[:, :, 1:]
                obs_hist_buf[:, :, -1] = frame
                obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float()

                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.control_dt - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopping policy execution...")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            # Disable motors safely
            self._driver.set_torque_mode(enable=False)
            print("Motors disabled.")

    def fetch_grasp_state(self):
        return torch.tensor(
            [
                [
                    0.000,
                    0.500,
                    0.000,
                    0.000,
                    -0.750,
                    1.300,
                    0.000,
                    0.750,
                    1.750,
                    1.500,
                    1.750,
                    1.750,
                    0.00,
                    1.0000,
                    0.0000,
                    0.00,
                ]
            ],
            device=self.device,
        )


if __name__ == "__main__":
    parent_path = Path(__file__).absolute().parent

    cfg_path = parent_path / "tasks/reorient_cube/config.yaml"
    ckpt_path = parent_path / "tasks/reorient_cube/ckpt.pth"

    servo_ids = list(range(16))
    driver = DynamixelDriver(
        servo_ids=servo_ids,
        port="/dev/cu.usbserial-FTA2U4SR",
        baud_rate=4_000_000,
        reading_interval=0.01,
    )

    controller = LeapHandController(driver)
    controller.restore_policy(
        policy=RLGamesPolicy(
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            num_proprio_obs=96,
            action_space=16,
            device="mps",
        )
    )
    controller.run()
