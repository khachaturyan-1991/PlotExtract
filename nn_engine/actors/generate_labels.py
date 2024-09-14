import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import cv2
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')


class NumberDataset(Dataset):
    def __init__(self, mode: str = "train", axis: str = "x", num_samples=10, transform=None):
        super(Dataset, self).__init__()
        self.num_samples = num_samples
        self.transform = transform
        self.image_paths = f"./data/labels/{mode}/{axis}/"
        self.df = pd.read_csv(self.image_paths + "labels.csv")

    def __len__(self):
        return self.num_samples

    def extract_number(self, filename):
        base = os.path.basename(filename)
        number = os.path.splitext(base)[0]
        return int(number)

    def __getitem__(self, idx):
        img = np.load(self.image_paths + f"{idx}.npy")
        img = np.expand_dims(img, axis=0)
        _, img = cv2.threshold(img, 0.9, 1, cv2.THRESH_BINARY)
        label = self.df.iloc[idx]
        label = torch.tensor([label[f"{i}"] for i in range(len(label) - 1)])
        return img, label


class GenerateSequenceDataset(Dataset):
    def __init__(self, num_samples, transform=None, tolerance: int = 50,
                 img_size: int = 128, fig_size: int = 5, dpi: int = 300):
        self.num_samples = num_samples
        self.transform = transform
        self.tolerance = tolerance
        self.dpi = dpi
        self.img_size = (img_size, img_size)
        self.fig_size = (fig_size, fig_size)

    def __len__(self):
        return self.num_samples

    def _encode_number(self, num):
        sign = 1 if num >= 0 else 0
        abs_num = abs(num)
        decimal_of_ten = abs_num // 10
        rest = abs_num % 10
        return [sign, int(decimal_of_ten), int(rest)]

    def __getitem__(self, idx):

        interval = np.random.choice([i for i in range(1, 50)])
        x = [-2 * interval, -1 * interval, 0, 1 * interval, 2 * interval]

        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        ax.set_xticks(x)
        ax.set_yticks(x)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(x), max(x))

        coef = np.random.uniform(-8, 8, 3)
        y = np.poly1d(coef)
        line_color1 = [0, 0, 1]
        ax.plot(x, y(x), color=line_color1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        image = image[:, :, :3]
        image = image.copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = F.interpolate(image.unsqueeze(0),
                              size=self.img_size,
                              mode='bilinear',
                              align_corners=False).squeeze(0)
        # Encode the first and last numbers
        first_label = self._encode_number(x[0])
        last_label = self._encode_number(x[-1])
        label = first_label + last_label
        img_x = np.expand_dims(image[2, 270: 270 + 20, :], axis=0)
        img_y = np.expand_dims(image[2, :, :30], axis=0)
        return torch.tensor(img_x,
                            dtype=torch.float32), torch.tensor(img_y,
                                                               dtype=torch.float32), torch.tensor(label)


def generate_data(mode: str = "train", num_samples: int = 1000,
                  img_size: int = 128, fig_size: int = 5, dpi: int = 300):
    if not os.path.exists(f"./data/labels/{mode}"):
        os.mkdir(f"./data/labels/{mode}")
    if not os.path.exists(f"./data/labels/{mode}/x") or not os.path.exists(f"./data/labels/{mode}/y"):
        os.mkdir(f"./data/labels/{mode}/x")
        os.mkdir(f"./data/labels/{mode}/y")
    dataset = GenerateSequenceDataset(num_samples=10, img_size=img_size, fig_size=fig_size, dpi=dpi)
    df = {}
    for i, (img_x, img_y, lable) in enumerate(dataset):
        df[i] = lable.numpy()
        np.save(f"./data/labels/{mode}/x/{i}.npy", img_x[0].numpy())
        np.save(f"./data/labels/{mode}/y/{i}.npy", img_x[0].numpy())
        if i == num_samples:
            break
    df = pd.DataFrame(df).T
    df.to_csv(f"./data/labels/{mode}/labels.csv")


if __name__ == "__main__":

    img_size = 296
    fig_size = 3
    dpi = 300
    generate_data(mode="train", num_samples=1792, img_size=img_size, fig_size=fig_size, dpi=dpi)
    generate_data(mode="test", num_samples=256, img_size=img_size, fig_size=fig_size, dpi=dpi)
    generate_data(mode="validation", num_samples=128, img_size=img_size, fig_size=fig_size, dpi=dpi)
