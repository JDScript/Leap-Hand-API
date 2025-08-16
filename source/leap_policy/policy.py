from pathlib import Path

from rl_games.algos_torch.model_builder import ModelBuilder
import torch
import yaml


def load_param_dict(cfg_path: Path):
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


class RLGamesPolicy:
    def __init__(self, cfg_path: str, num_proprio_obs, action_space, ckpt_path=None, device="cuda"):
        self.cfg_path = Path(cfg_path)
        self.ckpt_path = ckpt_path
        self.device = device

        # read the yaml file
        network_params = load_param_dict(cfg_path)["params"]

        # build the model config
        normalize_value = network_params["config"]["normalize_value"]
        normalize_input = network_params["config"]["normalize_input"]
        model_config = {
            "actions_num": action_space,
            "input_shape": (num_proprio_obs,),
            "num_seqs": 2,
            "value_size": 1,
            "normalize_value": normalize_value,
            "normalize_input": normalize_input,
        }

        # build the model
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config)

        # load checkpoint if available
        if self.ckpt_path is not None:
            weights = torch.load(self.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
            self.model.load_state_dict(weights["model"])
            if normalize_input and "running_mean_std" in weights:
                self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        # 转为float32 兼容MPS
        for param in self.model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        for _, buf in self.model.named_buffers():
            if buf.dtype != torch.float32:
                buf.data = buf.data.float()
        self.model = self.model.to(self.device)

        if self.model.is_rnn():
            hidden_states = self.model.get_default_rnn_state()
            self.hidden_states = [s.to(self.device).float() for s in hidden_states]

        # dummy variable
        self.dummy_prev_actions = torch.zeros((2, action_space), dtype=torch.float32).to(self.device)

    def step(self, proprio):
        batch_dict = {"is_train": False, "obs": proprio.repeat(2, 1), "prev_actions": self.dummy_prev_actions}
        if self.model.is_rnn():
            batch_dict["rnn_states"] = self.hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None

        # step through model
        res_dict = self.model(batch_dict)
        mus = res_dict["mus"][0:1]
        sigmas = res_dict["sigmas"][0:1]

        if "rnn_states" in res_dict:
            self.hidden_states = res_dict["rnn_states"]

        position = None

        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample()

        return {"mus": mus, "sigmas": sigmas, "obj_pose": position, "selected_action": selected_action}

    def reset_hidden_state(self):
        for i in range(len(self.hidden_states)):
            self.hidden_states[i] *= 0.0


def main():
    num_proprio_obs = 96
    action_space = 16

    parent_path = Path(__file__).absolute().parent

    cfg_path = parent_path / "tasks/reorient_cube/config.yaml"
    ckpt_path = parent_path / "tasks/reorient_cube/ckpt.pth"

    # create the model
    policy = RLGamesPolicy(
        cfg_path=cfg_path,
        num_proprio_obs=num_proprio_obs,
        action_space=action_space,
        ckpt_path=ckpt_path,
        device="mps",
    )

    dummy_proprio = torch.randn(1, num_proprio_obs, dtype=torch.float32).to("mps")
    policy.reset_hidden_state()

    # forward
    policy_out = policy.step(
        proprio=dummy_proprio,
    )

    # mus: [1, action_space]
    # sigmas: [1, action_space]
    # obj_pos: [1, 3]
    # selected_action: [1, action_space]

    print(policy_out)


if __name__ == "__main__":
    main()
