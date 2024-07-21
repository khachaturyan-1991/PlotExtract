import numpy as np
import torch
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')


class GenerateDataset(Dataset):
    def __init__(self, num_samples, transform=None, tolerance=150, img_size=128, fig_size=5):
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
        # Generate random coefficients
        a1, a2 = np.random.uniform(-10, 10, 2)
        a3, a4, a5 = np.random.uniform(-1, 1, 3)

        x = np.linspace(-10, 10, 100)

        y_linear = a1 * x + a2
        y_parabolic = a3 * x**2 + a4 * x + a5

        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)
        # np.random.rand(3,)
        line_color = [0.0, 0.0, 1.0]
        ax.plot(x, y_linear, color=line_color)
        parab_color = [0.0, 1.0, 0.0]
        ax.plot(x, y_parabolic, color=parab_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Convert RGBA to RGB
        image = image[:, :, :3]

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[image[:, :, 0] == 0] = 1
        mask[self.within_tolerance(image, line_color, self.tolerance)] = 2
        # mask[self.within_tolerance(image, parab_color, self.tolerance)] = 3

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        image = F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode='nearest').squeeze(0).squeeze(0).long()

        return image, mask


def generate_data(mode: str = "train", num_samples: int = 1000, img_size: int = 128, fig_size: int = 5):
    dataset = GenerateDataset(num_samples=num_samples, img_size=img_size, fig_size=fig_size)
    if not os.path.exists(f"./{mode}"):
        os.mkdir(f"./{mode}")
        os.mkdir(f"./{mode}/image")
        os.mkdir(f"./{mode}/mask")
    i = 0
    for img, mask in dataset:
        np.save(f'./{mode}/image/{i}.npy', img)
        np.save(f'./{mode}/mask/{i}.npy', mask)
        i += 1
        if i == num_samples:
            break
    print(f"{i + 1} images were saved to {mode}")


class LoadDataset(Dataset):
    def __init__(self, mode: str = "train", num_samples=10, transform=None, img_size: int = 128):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_size = (img_size, img_size)
        self.image_paths = glob.glob(f"./data/{mode}/image/*.npy")[:num_samples]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.load(img_path)
        mask_path = img_path.replace("image", "mask")
        mask = np.load(mask_path)
        _, image = cv2.threshold(image, 0.7, 1, cv2.THRESH_BINARY)
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
    import os
    img_size = 128
    generate_data(mode="train", num_samples=1280, img_size=img_size, fig_size=2)
    generate_data(mode="test", num_samples=128, img_size=img_size, fig_size=2)
    generate_data(mode="validation", num_samples=128, img_size=img_size, fig_size=2)

    # dataloader = create_dataloader(img_size=512)

    # for images, masks in dataloader:
    #     print(images.shape, masks.shape)
    #     break

    # print("Individual image size: ", images.shape, images.max(), images.min())
    # print("Corresponding mask size: ", masks.shape, masks.max(), masks.min())

    # n = 2
    # fig, axes = plt.subplots(4, n, figsize=(9, 5))
    # axes = axes.ravel()
    # for i in range(n):
    #     axes[i].imshow(images[i].permute(1, 2, 0))
    #     axes[i].axis("off")
    #     axes[i + n].imshow(255 * masks[i])
    #     axes[i + n].axis("off")
    #     axes[i + n].set_title(f"{np.unique(masks[i])}")

    # for i in range(n):
    #     axes[i + 2 * n].imshow(images[i + 6].permute(1, 2, 0))
    #     axes[i + 2 * n].axis("off")
    #     axes[i + 3 * n].imshow(255 * masks[i + 6])
    #     axes[i + 3 * n].axis("off")
    #     axes[i + 3 * n].set_title(f"{np.unique(masks[i + 6])}")

    # plt.tight_layout()
    # plt.savefig("generated_example.png")

    # fig = plt.figure()
    # plt.imshow(images[i + 6].permute(1, 2, 0))
    # plt.tight_layout()
    # plt.savefig("torm.png")
