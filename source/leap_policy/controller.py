from pathlib import Path
import time

import numpy as np
import torch

from source.leap.api import LeapHand
from source.leap.dynamixel.driver import DynamixelDriver
from source.leap_policy.policy import RLGamesPolicy

from .constants import REAL_TO_SIM_MAPPING
from .constants import SIM_LOWER_LIMITS
from .constants import SIM_TO_REAL_MAPPING
from .constants import SIM_UPPER_LIMITS
from .utils import saturate
from .utils import unscale


class LeapHandController(LeapHand):
    def __init__(self, driver, control_freq: int = 10):
        super().__init__(driver)
        self.policy: RLGamesPolicy | None = None
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq

        self.device = "mps"

        self.sim_dof_lower = torch.tensor(SIM_LOWER_LIMITS, device=self.device)
        self.sim_dof_upper = torch.tensor(SIM_UPPER_LIMITS, device=self.device)

        self.init_pose = torch.tensor(
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
            ],
            device=self.device,
        )
        self.action_scale = 1 / 24
        self.action_type = "relative"

        self.hist_len = 3

    def restore_policy(self, policy: RLGamesPolicy):
        self.policy = policy
        self.policy.reset_hidden_state()

    def poll_joint_position(self):
        "Returns the same joint position as isaac sim's physical dof position"
        joint_position = self._driver.get_joint_positions()
        joint_position = torch.from_numpy(joint_position.astype(np.float32)).to(device=self.device)
        joint_position -= torch.pi  # Shift to match sim's zero position
        joint_position = joint_position[REAL_TO_SIM_MAPPING]

        return {"position": joint_position}

    def forward_network(self, obs):
        return self.policy.step(obs)["selected_action"]

    def command_joint_position(self, sim_target: torch.Tensor):
        real_target = sim_target[SIM_TO_REAL_MAPPING]
        real_target += torch.pi
        self._driver.set_joint_positions(real_target.cpu().numpy())

    def run(self):
        assert self.policy is not None, "Policy must be restored before running"

        print("Command to the initial position")
        for _ in range(self.control_freq * 4):
            self.command_joint_position(self.init_pose)
            robot_state = self.poll_joint_position()
            obs = robot_state["position"]
            time.sleep(self.control_dt)
        print("Initial position reached!")

        # Get current state
        self.command_joint_position(self.init_pose)
        robot_state = self.poll_joint_position()
        obs = robot_state["position"]

        obs_hist_buf = torch.zeros((1, 32, self.hist_len), device=self.device, dtype=torch.float32)
        prev_target = obs.clone()

        unscaled_pos = unscale(obs, self.sim_dof_lower, self.sim_dof_upper)
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
                    target = prev_target + self.action_scale * action
                    target = saturate(target, self.sim_dof_lower, self.sim_dof_upper)
                else:
                    raise ValueError(f"Unsupported action type: {self.action_type}. Must be relative or absolute.")

                prev_target = target.clone()

                print(f"Sending command: {target}")
                self.command_joint_position(target)
                # self.command_joint_position(self.init_pose)

                robot_state = self.poll_joint_position()
                print(f"Received state: {robot_state['position']}")

                obs = robot_state["position"]
                unscaled_pos = unscale(obs, self.sim_dof_lower, self.sim_dof_upper)

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
