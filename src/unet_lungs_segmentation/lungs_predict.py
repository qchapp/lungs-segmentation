import numpy as np
import torch
from unet_lungs_segmentation.model import UNet
from torchvision.transforms import ToTensor
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from unet_lungs_segmentation.utils import get_weights


class LungsPredict:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = UNet(n_channels=1, n_class=1)
        self.checkpoint = get_weights(self.device)
        self.model.load_state_dict(self.checkpoint)
        self.model.to(self.device)

    """
        Predict the lungs segmentation of a given image by passing it to the trained model.
    """
    def predict(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self.preprocess(image).to(self.device)

        out = self.model(image_tensor)
        out = out.cpu().detach().numpy()
        out = np.squeeze(out)
        out = np.transpose(out, axes=(1, 2, 0))
        return resize(out, image.shape, order=0)

    """
        Apply some pre-processing before passing a given image to the model.
    """
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        image = resize(image, (128, 128, 128), order=0)
        image = rescale_intensity(image, out_range=(0, 1))
        image_tensor = ToTensor()(image)
        image_tensor = image_tensor[None]
        return image_tensor[None]

    """
        Apply a threshold of 0.5 to the segmented image in order to have a binary mask.
    """
    def postprocess(self, out: np.ndarray, threshold=0.5) -> np.ndarray:
        return out > threshold
    
    """
        Predict and then post-process (threshold) to obtain a binary mask.
    """
    def segment_lungs(self, image: np.ndarray, threshold=0.5) -> np.ndarray:
        raw_prediction = self.predict(image)
        binary_mask = self.postprocess(raw_prediction, threshold)
        return binary_mask