from optimum.exporters.onnx import main_export
from pathlib import Path

main_export(
    model_name_or_path="EleutherAI/gpt-neo-125M",
    output=Path("model"),
    task="text-generation",
    device="cpu"
)
