import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


class PlotDataset(Dataset):
    def __init__(self, num_samples, transform=None, tolerance=5):
        self.num_samples = num_samples
        self.transform = transform
        self.tolerance = tolerance

    def __len__(self):
        return self.num_samples

    def within_tolerance(self, img, color, tol):
        return np.all(np.abs(img - np.array(color) * 255) <= tol, axis=-1)
        
    def __getitem__(self, idx):

        a1, a2 = np.random.uniform(-10, 10, 2)
        a3, a4, a5 = np.random.uniform(-1, 1, 3)

        x = np.linspace(-10, 10, 100)

        # coeficients
        y_linear = a1 * x + a2
        y_parabolic = a3 * x**2 + a4 * x + a5

        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-100, 100)

        line_color = np.random.rand(3,)
        parab_color = np.random.rand(3,)
        ax.plot(x, y_linear, color=line_color)
        ax.plot(x, y_parabolic, color=parab_color)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # take only RGB
        image = image[:, :, :3]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[image[:, :, 0] == 0] = 1
        mask[self.within_tolerance(image, line_color, self.tolerance)] = 2
        mask[self.within_tolerance(image, parab_color, self.tolerance)] = 3

        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PlotDataset(num_samples=1000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, masks in dataloader:
        print(images.shape, masks.shape)
        break

    fig, axes = plt.subplots(4, 6, figsize=(9, 5))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i].permute(1,2,0))
        axes[i].axis("off")
        axes[i+6].imshow(masks[i])
        axes[i+6].axis("off")
        axes[i + 6].set_title(f"{np.unique(masks[i])}")

    for i in range(6):
        axes[i + 12].imshow(images[i + 6].permute(1,2,0))
        axes[i + 12].axis("off")
        axes[i + 18].imshow(masks[i + 6])
        axes[i + 18].axis("off")
        axes[i + 18].set_title(f"{np.unique(masks[i + 6])}")

    plt.tight_layout()
    plt.savefig("generated_example.png")
