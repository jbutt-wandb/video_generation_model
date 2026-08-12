import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    # Get specs of the device - nvidia smi type
    print(torch.cuda.get_device_properties(device))