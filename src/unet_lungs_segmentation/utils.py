import torch
import requests
from io import BytesIO

ZENODO_ID = "15011174"
FILE_NAME = "model_weights.pt"
url = f"https://zenodo.org/record/{ZENODO_ID}/files/{FILE_NAME}?download=1"

def get_weights():
    response = requests.get(url)
    buffer = BytesIO(response.content)
    return torch.load(buffer)