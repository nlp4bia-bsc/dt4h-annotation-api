import torch

REGISTRY_PATH = "app/model_manager/toy_registry.yaml"
RESOURCES_PATH = "app/resources"

def get_device():
    if not torch.cuda.is_available():
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability()
        capability = float(f"{major}.{minor}")

        # PyTorch build supports >= 7.0 in this codebase
        MIN_CAPABILITY = 7.0

        if capability < MIN_CAPABILITY:
            print(
                f"CUDA device found (capability {capability}) but not supported "
                f"(requires >= {MIN_CAPABILITY}). Falling back to CPU."
            )
            return "cpu"

        return "cuda"

    except Exception as e:
        print(f"CUDA check failed: {e}. Falling back to CPU.")
        return "cpu"
    
device = get_device()   