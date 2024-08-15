import numpy as np
import torch
import glob
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')


class GenerateDataset(Dataset):
    def __init__(self, num_samples, transform=None, tolerance=50, img_size=128, fig_size=5):
        self.num_samples = num_samples
        self.transform = transform
        self.tolerance = tolerance
        self.img_size = (img_size, img_size)
        self.fig_size = (fig_size, fig_size)

    def __len__(self):
        return self.num_samples

    def within_tolerance(self, img, color, tol):
        return np.all(np.abs(img - np.array(color) * 255) <= tol, axis=-1)

    def __getitem__(self, idx):

        x = np.linspace(-2, 2, 100)

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-10, 10)

        axes_color = [0.0, 0.0, 0.0]

        coef = np.random.uniform(-8, 8, 3)
        y = np.poly1d(coef)
        line_color1 = [0, 0, 1]
        ax.plot(x, y(x), color=line_color1)
        coef = np.random.uniform(-8, 8, 3)
        y = np.poly1d(coef)
        line_color2 = [0, 1, 0]
        ax.plot(x, y(x), color=line_color2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        image = image[:, :, :3]

        axes_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        plot_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        axes_mask[self.within_tolerance(image, axes_color, self.tolerance)] = 1
        plot_mask[self.within_tolerance(image, line_color1, self.tolerance)] = 1
        plot_mask[self.within_tolerance(image, line_color2, self.tolerance)] = 1

        mask = np.stack((axes_mask, plot_mask), axis=0)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        image = F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.img_size, mode='nearest').squeeze(0).long()

        return image, mask, coef


def generate_data(mode: str = "train", num_samples: int = 1000, img_size: int = 128, fig_size: int = 5):
    dataset = GenerateDataset(num_samples=num_samples, img_size=img_size, fig_size=fig_size)
    if not os.path.exists(f"./{mode}"):
        os.mkdir(f"./{mode}")
        os.mkdir(f"./{mode}/image")
        os.mkdir(f"./{mode}/mask")
    dataset = GenerateDataset(num_samples=num_samples, img_size=img_size, fig_size=fig_size)
    coefs = np.zeros((num_samples, 3))
    i = 0
    for img, mask, coef in dataset:
        np.save(f'./{mode}/image/{i}.npy', img)
        np.save(f'./{mode}/mask/{i}.npy', mask)
        coefs[i] = coef
        i += 1
        if i == num_samples:
            break
    print(f"{i} images were saved to {mode}")
    np.save(f'./{mode}/coefs.npy', coefs)


class LoadDataset(Dataset):
    def __init__(self, mode: str = "train", num_samples=10, transform=None, img_size: int = 128):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_size = (img_size, img_size)
        image_paths = glob.glob(f"./data/{mode}/image/*.npy")[:num_samples]
        self.image_paths = sorted(image_paths, key=self.extract_number)

    def __len__(self):
        return len(self.image_paths)

    def extract_number(self, filename):
        base = os.path.basename(filename)
        number = os.path.splitext(base)[0]
        return int(number)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.load(img_path)
        mask_path = img_path.replace("image", "mask")
        mask = np.load(mask_path)
        _, image = cv2.threshold(image, 0.9, 1, cv2.THRESH_BINARY)
        return image, mask


def create_dataloader(num_samples: int = 32,
                      batch_size: int = 32,
                      shuffle: bool = False,
                      img_size: int = 128,
                      mode: str = "train"
                      ):
    dataset = LoadDataset(mode=mode, num_samples=num_samples, img_size=img_size)
    print(dataset.__len__())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":

    img_size = 128
    generate_data(mode="train", num_samples=1024, img_size=img_size, fig_size=2)
    generate_data(mode="test", num_samples=128, img_size=img_size, fig_size=2)
    generate_data(mode="validation", num_samples=128, img_size=img_size, fig_size=2)
