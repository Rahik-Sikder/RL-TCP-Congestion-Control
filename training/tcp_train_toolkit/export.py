import json
import os
from datetime import datetime

import torch


def export_onnx(model, model_name: str, input_size: int, device, k: int = None) -> str:
    """Export model to ONNX inside training/outputs/{model_name}_{date}_{time}/.

    Also writes {model_name}.info.json with {"k": k} when k is provided.
    Returns the path to the output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        f"{model_name}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    dummy_input = torch.randn(1, input_size).to(device)

    model.eval()
    model_key = model_name.lower()
    if model_key == "ppo":
        output_names = ["action_mean", "action_std", "value"]
    elif model_key == "dqn":
        output_names = ["q_values"]
    elif model_key == "ddpg":
        output_names = ["action"]
    else:
        output_names = ["output"]

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["state"], output_names=output_names,
        dynamo=False,
    )

    if k is not None:
        info_path = os.path.join(output_dir, f"{model_name}.info.json")
        with open(info_path, "w") as f:
            json.dump({"k": k}, f, indent=4)

    return output_dir
