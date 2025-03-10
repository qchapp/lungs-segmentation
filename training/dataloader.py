import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import tifffile
from skimage.transform import resize
from skimage.exposure import rescale_intensity

class CSVReader():
    def __init__(self, file_name):
        self.file = pd.read_csv(file_name)
        self.length = len(self.file)

    # load only data for the training set
    def training_data(self):
        to_drop = []
        for i in range(self.length):
            if self.file.iloc[i, 1] == "no" or (self.file.iloc[i, 2] == "no" and self.file.iloc[i, 3] == "no"):
                to_drop.append(i)
        return self.file.drop(index=to_drop)
    
    # load only data for the validation set
    def validation_data(self):
        to_drop = []
        for i in range(self.length):
            if self.file.iloc[i, 1] == "yes" or (self.file.iloc[i, 2] == "no" and self.file.iloc[i, 3] == "no"):
                to_drop.append(i)
        return self.file.drop(index=to_drop)
    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, is_training=True):
        if is_training:
            self.img_labels = CSVReader(annotations_file).training_data()
        else:
            self.img_labels = CSVReader(annotations_file).validation_data()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], "image.tif")
        image = tifffile.imread(img_path).astype(np.float32)
        image = resize(image, (128, 128, 128), order=0)
        image = rescale_intensity(image, out_range=(0, 1))

        if self.img_labels.iloc[idx, 2] == "yes":
            label = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], "lungs.tif")
        else:
            label = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], "lungs2.tif")

        label = tifffile.imread(label).astype(np.float32)
        label = resize(label, (128, 128, 128), order=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = image[None]
        label = label[None]


        return image, label