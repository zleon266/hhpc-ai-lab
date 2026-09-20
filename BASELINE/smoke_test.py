import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA is unavailable"

name = torch.cuda.get_device_name(0)
print("GPU:", name)

assert "A100" in name, f"Expected A100, got: {name}"

print("A100 smoke test PASSED")
