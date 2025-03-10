import sys
from os import listdir
from os.path import join, isdir, exists
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

import run_ilastik
import ilastik


""" 
    Saves all the segmented lungs from specified folder in a tif file and their corresponding png plot for manual check. Run it from command line.
"""
if __name__ == "__main__":
    _, data_folder = sys.argv
    data_folder = Path(data_folder).resolve()

    nb_images = 0 # counter

    animals = [Path(join(data_folder, f)).resolve() for f in listdir(data_folder) if isdir(join(data_folder, f))] # list of paths of different animals directory
    for animal in animals: # list differents animals
        for frame in listdir(animal): # list different frames for an animal
            frame = Path(join(animal, frame)).resolve()
            image_file = Path(join(frame, "image.tif")).resolve()
            if exists(Path(join(frame, "lungs.tif")).resolve()): continue

            lungs_hull = run_ilastik.lungs_hull_seg(image_file) # segmentation on the current frame
            
            if lungs_hull.sum(): # fast check that the segmentation worked
                tifffile.imwrite(frame / "lungs.tif", lungs_hull) # we save the segmentation through a tif file in the folder
                nb_images += 1

            fig_path = image_file.parent / "lungs.png"
            image = tifffile.imread(str(image_file))
            fig = ilastik.volume_slice_figure(lungs_hull, image, skip_level=8)
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
    
    print(f"Done to {nb_images} images!")