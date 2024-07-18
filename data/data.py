import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')


class PlotDataset(Dataset):
    def __init__(self, num_samples, transform=None, tolerance=5, img_size=(128, 128)):
        self.num_samples = num_samples
        self.transform = transform
        self.tolerance = tolerance
        self.img_size = img_size  # Image size in pixels (width, height)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random coefficients
        a1, a2 = np.random.uniform(-10, 10, 2)
        a3, a4, a5 = np.random.uniform(-1, 1, 3)

        # Generate x values
        x = np.linspace(-10, 10, 100)

        # Generate y values for linear and parabolic functions
        y_linear = a1 * x + a2
        y_parabolic = a3 * x**2 + a4 * x + a5

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))  # Control the figure size here
        ax.set_xlim(-10, 10)
        ax.set_ylim(-100, 100)

        # Plot the linear function
        line_color = np.random.rand(3,)
        ax.plot(x, y_linear, color=line_color)

        # Plot the parabolic function
        parab_color = np.random.rand(3,)
        ax.plot(x, y_parabolic, color=parab_color)

        # Save plot as image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)  # Close the figure to avoid displaying it

        # Convert RGBA to RGB
        image = image[:, :, :3]

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw axes on mask (Assuming axes are black lines)
        mask[image[:, :, 0] == 0] = 1

        # Define a helper function for color matching with tolerance
        def within_tolerance(img, color, tol):
            return np.all(np.abs(img - np.array(color) * 255) <= tol, axis=-1)

        # Add linear function to mask
        mask[within_tolerance(image, line_color, self.tolerance)] = 2

        # Add parabolic function to mask
        mask[within_tolerance(image, parab_color, self.tolerance)] = 3

        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        # Resize image and mask
        image = F.interpolate(image.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode='nearest').squeeze(0).squeeze(0).long()

        return image, mask


def create_dataloader(num_samples: int = 32, batch_size: int = 32, shuffle: bool = False, img_size: tuple = (128, 128)):
    dataset = PlotDataset(num_samples=num_samples, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":

    dataloader = create_dataloader()

    for images, masks in dataloader:
        print(images.shape, masks.shape)
        break

    print("Individual image size: ", images.shape, images.max(), images.min())
    print("Corresponding mask size: ", masks.shape, masks.max(), masks.min())

    fig, axes = plt.subplots(4, 6, figsize=(9, 5))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].axis("off")
        axes[i + 6].imshow(255 * masks[i])
        axes[i + 6].axis("off")
        axes[i + 6].set_title(f"{np.unique(masks[i])}")

    for i in range(6):
        axes[i + 12].imshow(images[i + 6].permute(1, 2, 0))
        axes[i + 12].axis("off")
        axes[i + 18].imshow(255 * masks[i + 6])
        axes[i + 18].axis("off")
        axes[i + 18].set_title(f"{np.unique(masks[i + 6])}")

    plt.tight_layout()
    plt.savefig("generated_example.png")
