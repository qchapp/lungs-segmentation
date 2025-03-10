import sys
import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import datetime

import run_ilastik

def volume_slice_figure(preds, image, skip_level=1):
    """
    
    """
    def color_alpha(arr, alpha=1):
        mask = arr == 0
        view = np.empty((arr.shape[0], arr.shape[1], 4))
        view[mask] = 0
        view[~mask] = [1, 0.4, 0.2, alpha]
        return view

    preds = preds[::skip_level]
    image = image[::skip_level]

    fig, axes = plt.subplots(nrows=len(preds) // 8, ncols=8, figsize=(16, 8), facecolor='white')
    for ax, preds_slice, image_slice in zip(axes.ravel(), preds, image):
        backviewfull = color_alpha(preds_slice, alpha=0.3)
        ax.imshow(image_slice, cmap=plt.cm.gray)
        ax.imshow(backviewfull)
        ax.margins(x=0, y=0)
        ax.axis('off')
    
    return fig
    

if __name__ == '__main__':
    _, image_file = sys.argv
    start = datetime.datetime.now()
    image_file = Path(image_file).resolve()

    # image_normed = run_normalize.normalize_image(image)

    lungs_hull = run_ilastik.lungs_hull_seg(image_file)

    fig_path = image_file.parent / f'{image_file.stem}_lungs2.png'
    image = tifffile.imread(str(image_file))
    fig = volume_slice_figure(lungs_hull, image, skip_level=16)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"\n{str(datetime.datetime.now() - start)}")