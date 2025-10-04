# 🐭 Lungs segmentation in mice CT scans

Neural network for segmenting mouse lungs in CT scans, based on the classic [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) architecture.

<p align="center">
  <img src="https://raw.githubusercontent.com/qchapp/lungs-segmentation/refs/heads/master/images/main_fig.png" height="500" alt="Model overview">
</p>

The goal is to provide a reliable, easy-to-use tool to obtain **binary lung masks** from mouse CT scans.

---

## Quick links

- [Try in your browser (Hugging Face Space)](#try-in-your-browser-hugging-face-space)
- [Installation](#installation)
- [Use in napari (GUI)](#usage-in-napari)
- [Use as a Python library](#usage-as-a-library)
- [Use from the command line](#usage-as-a-cli)
- [Run with Docker (GHCR)](#run-with-docker-ghcr)
- [Models & weights](#models)
- [Dataset](#dataset)
- [Issues](#issues)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Carbon footprint](#carbon-footprint-of-this-project)

---

## Try in your browser (Hugging Face Space)

No local installation needed — test the model directly in your browser:

➡️ **https://huggingface.co/spaces/qchapp/3d-lungs-segmentation**

Upload a mouse CT scan, run the segmentation, visualize and download the resulting lung mask.

---

## Installation

We recommend using a fresh Python environment.

- **Requirements:** `python>=3.9`, `pytorch>=2.0`  
  Please install PyTorch first for your platform following the instructions on [pytorch.org](https://pytorch.org/get-started/locally/).

Install the package from PyPI:

```bash
pip install unet_lungs_segmentation
```

Or install from source:

```bash
git clone https://github.com/qchapp/lungs-segmentation.git
cd lungs-segmentation
pip install -e .
```

---

## Usage in napari

[Napari](https://napari.org/stable/) is a multidimensional image viewer for Python.

1. Launch napari:

   ```bash
   napari
   ```

2. Open an image via **File → Open files…** or drag & drop into the viewer.  
   To open formats like **NIfTI** directly, consider installing the plugin
   [`napari-medical-image-formats`](https://pypi.org/project/napari-medical-image-formats/).

3. **Sample data:** In napari, go to **File → Open Sample → Mouse lung CT scan** to try the model.

4. Run the plugin via **Plugins → Lungs segmentation (unet_lungs_segmentation)**.  
   Select the image layer and click **Segment lungs**.

<p align="center">
  <img src="https://raw.githubusercontent.com/qchapp/lungs-segmentation/refs/heads/master/images/napari-screenshot.png" height="500" alt="napari plugin screenshot">
</p>

---

## Usage as a library

Run the model in a few lines to obtain a binary mask (NumPy array):

```python
from unet_lungs_segmentation import LungsPredict

lungs_predict = LungsPredict()
mask = lungs_predict.segment_lungs(your_image)           # your_image: NumPy array
```

Specify a custom probability `threshold` (float in `[0, 1]`) if desired:

```python
mask = lungs_predict.segment_lungs(your_image, threshold=0.5)
```

---

## Usage as a CLI

Run inference on a single image:

```bash
uls_predict_image -i /path/to/folder/image_001.tif [-t <threshold>]
```

- If `-t`/`--threshold` is provided, it binarizes the prediction at that value (default: `0.5` in `[0,1]`).
- The output mask is saved next to the input image:

```
folder/
├── image_001.tif
└── image_001_mask.tif
```

Batch-process a folder:

```bash
uls_predict_folder -i /path/to/folder/ [-t <threshold>]
```

Produces, e.g.:

```
folder/
├── image_001.tif
├── image_001_mask.tif
├── image_002.tif
└── image_002_mask.tif
```

---

## Run with Docker (GHCR)

You can run the tool without a local Python setup using our container on **GitHub Container Registry**.

1. **Pull the image:**

```bash
docker pull ghcr.io/qchapp/unet-lungs-segmentation:latest
```

2. **Run on a single image** (mount a data folder into the container):

```bash
docker run --rm -v /path/to/data:/data ghcr.io/qchapp/unet-lungs-segmentation:latest \
  -i /data/image_001.tif -t 0.5
```

3. **Run on a folder:**

```bash
docker run --rm -v /path/to/data:/data ghcr.io/qchapp/unet-lungs-segmentation:latest \
  uls_predict_folder -i /data
```

> **Note:** The container’s entrypoint is `uls_predict_image`.  
> For folder mode, prefix the command explicitly as shown above.

---

## Models

The model weights (~1 GB) are downloaded automatically on first use from
[Hugging Face](https://huggingface.co/qchapp/unet-lungs-segmentation-weights).

---

## Dataset

The model was trained on **355** images from 17 different experiments and 2 scanners, and validated on **62** images.

---

## Issues

If you encounter problems, please open an issue with a clear description and, if possible, a minimal reproducible example.

---

## Acknowledgments

This project was developed as part of a **Bachelor’s project** at the *EPFL Center for Imaging*.  
It was carried out under the supervision of **Mallory Wittwer** and **Edward Andò**, whom we sincerely thank for their guidance and support.

---

## License

This model is released under the [BSD-3](LICENSE.txt) license.

---

## Carbon footprint of this project

As estimated by [Green Algorithms](http://calculator.green-algorithms.org/), training this model emitted approximately **584 g CO₂e**.
