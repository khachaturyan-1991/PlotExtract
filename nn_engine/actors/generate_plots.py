import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')


class GenerateDataset(Dataset):

    def __init__(self, num_samples, transform=None, tolerances: list = [80, 30],
                 img_size: int = 128, fig_size: int = 5, dpi: int = 300):
        self.num_samples = num_samples
        self.transform = transform
        self.tolerance = tolerances
        self.dpi = dpi
        self.img_size = (img_size, img_size)
        self.fig_size = (fig_size, fig_size)

    def __len__(self):
        return self.num_samples

    def within_tolerance(self, img, color, tol):
        return np.all(np.abs(img - np.array(color) * 255) <= tol, axis=-1)

    def create_mask_with_squares(self, image, square_size=15):
        # Create a blank mask of the same size as the input image
        mask = np.zeros_like(image, dtype=np.uint8)

        # Find connected components (labels each connected group of pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        # Calculate the half size of the square to center it on each tick/number
        half_size = square_size  # // 2

        # Loop through each centroid and draw a square
        for (x, y) in centroids[1:]:  # Skipping the first centroid (background)
            x, y = int(x), int(y)
            top_left = (max(0, x - half_size), max(0, y - half_size))
            bottom_right = (min(mask.shape[1], x + half_size), min(mask.shape[0], y + half_size))
            cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), thickness=-1)

        return mask // 255

    def __getitem__(self, idx):
        x = np.linspace(-2, 2, 100)

        fig, ax = plt.subplots(figsize=self.fig_size, dpi=300)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-10 + 2 * i for i in range(11)])
        ax.set_ylim(-10, 10)

        axes_color = [0.0, 0.0, 0.0]
        axes_labels = [1.0, 0.0, 0.0]

        N_OF_PLOTS = 2
        coefs, line_colors = {}, {}
        for i in range(N_OF_PLOTS):
            coefs[i] = np.random.uniform(-8, 8, 3)
            y = np.poly1d(coefs[i])
            line_colors[i] = np.random.uniform(0, 1, 3).round(2)
            ax.plot(x, y(x), color=line_colors[i])

        ax.tick_params(axis='x', colors='red')
        ax.tick_params(axis='y', colors='red')
        ax.tick_params(axis='x', which='both', color='blue')
        ax.tick_params(axis='y', which='both', color='blue')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        image = image[:, :, :3]

        axes_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        label_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        plot_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        axes_mask[self.within_tolerance(image, axes_color, self.tolerance[0])] = 1
        label_mask[self.within_tolerance(image, axes_labels, self.tolerance[0])] = 1
        label_mask = self.create_mask_with_squares(label_mask)
        for i in range(N_OF_PLOTS):
            plot_mask[self.within_tolerance(image, line_colors[i], self.tolerance[1])] = 1

        mask = np.stack((label_mask, plot_mask), axis=0)  # np.stack((axes_mask, label_mask, plot_mask), axis=0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).long()

        image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.img_size, mode='nearest').squeeze(0).long()

        return image, mask, coefs


def generate_data(mode: str = "train", axis: str = "x", num_samples: int = 1000,
                  img_size: int = 128, fig_size: int = 5, dpi: int = 300):
    if not os.path.exists(f"./data/plots/{mode}"):
        os.mkdir(f"./data/plots/{mode}")
        os.mkdir(f"./data/plots/{mode}/image")
        os.mkdir(f"./data/plots/{mode}/mask")
    dataset = GenerateDataset(num_samples=num_samples, img_size=img_size, fig_size=fig_size, dpi=dpi)
    coefs = np.zeros((num_samples, 2, 3))
    i = 0
    for img, mask, coef in dataset:
        np.save(f'./data/plots/{mode}/image/{i}.npy', img)
        np.save(f'./data/plots/{mode}/mask/{i}.npy', mask)
        coefs[i] = [coef[key]for key in coef.keys()]  # coef
        i += 1
        if i == num_samples:
            break
    print(f"{i} images were saved to {mode}")
    np.save(f'./data/plots/{mode}/coefs.npy', coefs)


if __name__ == "__main__":

    img_size = 296
    fig_size = 3
    dpi = 300
    generate_data(mode="train", num_samples=1280, img_size=img_size, fig_size=fig_size, dpi=dpi)
    generate_data(mode="test", num_samples=128, img_size=img_size, fig_size=fig_size, dpi=dpi)
    generate_data(mode="validation", num_samples=12, img_size=img_size, fig_size=fig_size, dpi=dpi)
