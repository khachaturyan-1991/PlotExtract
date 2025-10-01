import numpy as np
import pandas as pd
import os
import cv2
import glob
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')


class NumberDataset(Dataset):
    def __init__(self, mode: str = "train", axis: str = "x", num_samples=10):
        super(Dataset, self).__init__()
        self.images = np.load(f"./data/labels/{mode}/images_{axis}.npy")
        self.labels = np.load(f"./data/labels/{mode}/labels.npy")
        self.num_samples = min(num_samples, self.images.shape[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        _, img = cv2.threshold(img, 0.9, 1, cv2.THRESH_BINARY)
        img = np.expand_dims(img, axis=0)
        label = self.labels[idx]
        label = torch.tensor(label)
        return img, label


class PlotImageLoader(Dataset):
    def __init__(self, mode: str = "train", num_samples=10, img_size: int = 128):
        super(Dataset, self).__init__()
        self.img_size = (img_size, img_size)
        self.images = np.load(f"./data/plots/{mode}/images.npy")
        self.masks = np.load(f"./data/plots/{mode}/masks.npy")
        self.num_samples = min(num_samples, self.images.shape[0])

    def __len__(self):
        return self.num_samples

    def extract_number(self, filename):
        base = os.path.basename(filename)
        number = os.path.splitext(base)[0]
        return int(number)

    def __getitem__(self, idx):
        mask = self.masks[idx]
        image = self.images[idx]
        _, image = cv2.threshold(image, 0.9, 1, cv2.THRESH_BINARY)
        return image, mask
