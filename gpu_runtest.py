import torch

"""To see if GPU is connected."""
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")