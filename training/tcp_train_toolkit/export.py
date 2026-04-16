import os
from datetime import datetime

import torch


def export_onnx(model, model_name: str, input_size: int, device) -> str:
    """Export model to ONNX inside training/outputs/{model_name}_{date}_{time}/.

    Returns the path to the exported .onnx file.
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
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["state"], output_names=["action_mean", "action_std", "value"],
        dynamo=False,
    )

    return onnx_path
