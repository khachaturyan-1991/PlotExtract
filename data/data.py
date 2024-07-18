import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')


class PlotDataset(Dataset):
    def __init__(self, num_samples: int = 100, tolerance: int = 5, img_size: int = 128):
        self.num_samples = num_samples
        self.tolerance = tolerance
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def smart_resize(self, img: np.array,
                     channel: int = 3):
        """
        Description
        -------
        Resize an image to squeeze into a template; \
        It should be used to resize images to a shape suitable \
        for neaural a neural network, \
        but preserving all the proportions of an object on an image

        Parameters
        -------
        img (ndarra): an image
        channel (int): number of channel, Default 3

        Return
        -------
        tamplate (ndarray): resized image
        """
        height, width = img.shape[:2]
        scale = min(self.img_size / width, self.img_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(img,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

        tamplate = np.zeros((self.img_size, self.img_size, channel),
                            dtype=np.uint8)
        x_offset = (self.img_size - new_width) // 2
        y_offset = (self.img_size - new_height) // 2

        tamplate[y_offset:y_offset + new_height,
                 x_offset:x_offset + new_width] = resized_image

        return tamplate

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
        image = self.smart_resize(image, )

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[image[:, :, 0] == 0] = 1
        mask[self.within_tolerance(image, line_color, self.tolerance)] = 2
        mask[self.within_tolerance(image, parab_color, self.tolerance)] = 3

        image = torch.tensor(image, dtype=torch.long).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long).permute(2, 0, 1)

        return image, mask


def create_dataloader(num_samples: int = 32, batch_size: int = 32, shuffle: bool = False, img_size: int = 128):
    dataset = PlotDataset(num_samples=num_samples, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":

    dataloader = create_dataloader()

    for images, masks in dataloader:
        print(images.shape, masks.shape)
        break

    fig, axes = plt.subplots(4, 6, figsize=(9, 5))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].axis("off")
        axes[i + 6].imshow(masks[i].permute(1, 2, 0))
        axes[i + 6].axis("off")
        axes[i + 6].set_title(f"{np.unique(masks[i])}")

    for i in range(6):
        axes[i + 12].imshow(images[i + 6].permute(1, 2, 0))
        axes[i + 12].axis("off")
        axes[i + 18].imshow(masks[i + 6].permute(1, 2, 0))
        axes[i + 18].axis("off")
        axes[i + 18].set_title(f"{np.unique(masks[i + 6])}")

    plt.tight_layout()
    plt.savefig("generated_example.png")
    print("Individual image size: ", images[0].shape)
    print("Corresponding mask size: ", masks[0].shape)
