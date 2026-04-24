import gymnasium as gym
import sys
sys.modules["gym"] = gym

import argparse
import json
import os
import torch
import torch.optim as optim

from tcp_train_toolkit.models.ppo_cnn import PpoActorCritic1DCNN
from tcp_train_toolkit.models.dqn_cnn import DqnQNetwork1DCNN
from tcp_train_toolkit import export, ppo_trainer, dqn_trainer

# Default hyperparameters — used as fallback when a key is absent from model_params.json
DEFAULTS = {
    "ppo": {
        "k": 10,
        "hidden_dim": 64,
        "lr": 3e-4,
        "num_episodes": 500,
        "gamma": 0.99,
        "port": 5555,
        "sim_seed": 42,
    },
    "dqn": {
        "k": 10,
        "hidden_dim": 64,
        "lr": 1e-4,
        "num_episodes": 500,
        "gamma": 0.99,
        "port": 5555,
        "sim_seed": 42,
        "batch_size": 64,
        "replay_capacity": 50000,
        "warmup_steps": 1000,
        "train_freq": 1,
        "target_update_freq": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 50000,
    },
}

MODEL_TRAINERS = {
    "ppo": ppo_trainer,
    "dqn": dqn_trainer,
}


def load_params(model: str) -> tuple:
    """Load and merge hyperparameters from model_params.json with DEFAULTS.

    Returns:
        (params, sim_list) where sim_list is None if not specified in the JSON.
    """
    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_params.json")

    json_params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            json_params = json.load(f)

    # Merge: defaults first, then any JSON overrides for this model
    params = {**DEFAULTS[model], **json_params.get(model, {})}

    # Resolve sim_list: use JSON value if present and non-empty, otherwise None
    json_sim_list = json_params.get("tcp_sim_list")
    sim_list = json_sim_list if json_sim_list else None

    return params, sim_list


def main():
    parser = argparse.ArgumentParser(description="Train a TCP congestion control RL agent in NS-3")
    parser.add_argument(
        "--model",
        default="ppo",
        help="RL algorithm / model architecture to train (ppo or dqn). Case-insensitive. Default: ppo",
    )
    args = parser.parse_args()
    model_name = args.model.lower()
    if model_name not in MODEL_TRAINERS:
        raise ValueError(f"Unknown model: {args.model}. Valid options: {list(MODEL_TRAINERS.keys())}")

    params, sim_list = load_params(model_name)

    # ns-3.40 lives alongside this script
    start_cwd = os.getcwd()
    ns3_dir = os.path.join(start_cwd, "ns-3.40")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ppo":
        model = PpoActorCritic1DCNN(k=params["k"], hidden_dim=params["hidden_dim"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    elif model_name == "dqn":
        model = DqnQNetwork1DCNN(k=params["k"], hidden_dim=params["hidden_dim"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainer = MODEL_TRAINERS[model_name]
    trainer.run(model, optimizer, ns3_dir, device, params, sim_list=sim_list)

    # Export to ONNX for Kathará C integration
    print("Training complete. Exporting model to ONNX...")
    state_dim = (params["k"] * 3) + 1
    onnx_path = export.export_onnx(model, model_name, state_dim, device, k=params["k"])
    print(f"Export saved to {onnx_path}")


if __name__ == "__main__":
    main()
