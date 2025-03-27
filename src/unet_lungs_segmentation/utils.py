from huggingface_hub import hf_hub_download
import torch

REPO_ID = "titi100/unet-lungs-segmentation-weights"
FILENAME = "weights.pt"

def get_weights(device):
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return torch.load(model_path, map_location=device)