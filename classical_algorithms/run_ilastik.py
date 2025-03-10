import os
import subprocess
from pathlib import Path
from csbdeep.utils import normalize

import numpy as np
import tifffile
from scipy.ndimage import label
from skimage.morphology import dilation, convex_hull_image

from dotenv import load_dotenv
load_dotenv()

ILASTIK_RUN_PATH = Path(os.getenv('ILASTIK_RUN_PATH'))
ILASTIK_PROJECT = Path('.').resolve() / os.getenv('ILASTIK_PROJECT')

def lungs_hull_seg(image_path: str):
    """
    Segmentation of the convex hull of the lungs using a trained Ilastik model
    """

    image = tifffile.imread(image_path)
    image_normed = normalize_image(image)

    image_path = Path(image_path)
    image_normed_path = image_path.parent / f"{image_path.stem}_normed.tif"
    tifffile.imwrite(image_normed_path, image_normed)

    ilastik_output_file = image_normed_path.parent / f"{image_normed_path.stem}_Simple Segmentation.npy"

    subprocess.run(
        [
            ILASTIK_RUN_PATH,
            "--headless",
            "--project",
            ILASTIK_PROJECT,
            "--export_source",
            "Simple Segmentation",
            image_normed_path,
        ]
    )

    try:
        ilastik_output_image = np.squeeze(np.load(ilastik_output_file))
    except:
        print("Could not load: ", ilastik_output_file)
        import pdb; pdb.set_trace()
        return

    # The lungs contour label index is 4
    lungs_contour = (ilastik_output_image == 4).astype(np.uint8)
    lungs_hull = lung_contour_to_hull(lungs_contour)

    os.remove(ilastik_output_file)
    os.remove(image_normed_path)

    return lungs_hull

def normalize_image(img):
    """
    Quantile-based image normalization.
    Note: Can we write an equivalent function with Numpy and get rid of the csbdeep dependency?
    """
    if (img.min() == 0.0) & (img.max() == 1.0):
        img_normed = img.astype(np.float32)
    else:
        img_normed = normalize(img, 2, 98, axis=(0, 1, 2))
    return img_normed

def keep_biggest_object(lab_int: np.ndarray) -> np.ndarray:
    """Selects only the biggest object of a labels image."""
    labels = label(lab_int)[0]  # label from scipy

    counts = np.unique(labels, return_counts=1)

    if len(counts) <= 1:
        return np.zeros_like(lab_int)
    else:
        biggestLabel = np.argmax(counts[1][1:]) + 1
        return (labels == biggestLabel).astype(int)


def lung_contour_to_hull(image: np.ndarray) -> np.ndarray:
    hull = np.zeros_like(image, dtype=np.uint8)
    image = keep_biggest_object(image)

    for k, z_layer in enumerate(image):
        if z_layer.sum():
            hull[k] = dilation(dilation(dilation(dilation(convex_hull_image(z_layer)))))

    return hull