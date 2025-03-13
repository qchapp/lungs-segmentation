from unet_lungs_segmentation import LungsPredict
import tifffile
from pathlib import Path
import argparse
import glob

def process_input_file_predict(input_image_file, threshold, lungs_predict):
    image = tifffile.imread(input_image_file)

    if threshold is None:
        threshold = 0.5
    pred = lungs_predict.segment_lungs(image, threshold)

    pt = Path(input_image_file)
    out_file_name = pt.parent / f"{pt.stem}_mask.tif"

    tifffile.imwrite(out_file_name, pred)
    print("Wrote to ", out_file_name)


def cli_predict_image():
    """Command-line entry point for model inference."""
    parser = argparse.ArgumentParser(description="Use this command to run inference.")
    parser.add_argument("-i", type=str, required=True, help="Input image. Must be either a TIF or a NIFTI image file.")
    parser.add_argument("-t", type=int, required=False, help="Threshold applied during postprocessing. Default to 0.5.")
    args = parser.parse_args()

    input_image_file = Path(args.i).resolve()

    if not input_image_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_image_file}\nTry to give an absolute path.")

    threshold = args.t

    lungs_predict = LungsPredict()

    process_input_file_predict(input_image_file, threshold, lungs_predict)


def cli_predict_folder():
    parser = argparse.ArgumentParser(description="Use this command to run inference in batch on a given folder.")
    parser.add_argument("-i", type=str, required=True, help="Input folder. Must contain suitable TIF image files.")
    parser.add_argument("-t", type=int, required=False, help="Threshold applied during postprocessing. Default to 0.5.")
    args = parser.parse_args()

    input_folder = Path(args.i).resolve()

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}\nTry to give an absolute path.")

    threshold = args.t

    lungs_predict = LungsPredict()

    for input_image_file in glob.glob(str(Path(input_folder) / "*.tif")):
        process_input_file_predict(input_image_file, threshold, lungs_predict)
