import numpy as np
import matplotlib.pylab as plt
import argparse
import os
import torch


def plot_hystory(h_train: dict,
                 h_test: dict,
                 file_name: str = "history"):
    _ = plt.figure(figsize=(6, 6))

    plt.plot(h_train.keys(),
             h_train.values(),
             "-o", label="Train")

    plt.plot(h_test.keys(),
             h_test.values(),
             "-*", label="Validation")

    plt.legend(title="Dataset")

    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.savefig(f"{file_name}.png")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action",
                        type=str, default="extract",
                        help="Choose between plot extraction (extract), models training (train), or data generation (data)")
    parser.add_argument("--num_of_samples",
                        type=int, nargs='+', default=[1280, 128, 128],
                        help="Choose number of images for [train, validation, test]")
    # data
    parser.add_argument("--img_size",
                        type=int, default=296,
                        help="Image size")
    parser.add_argument("--fig_size",
                        type=int, default=5,
                        help="Plots size (plot on an image)")
    parser.add_argument("--dpi",
                        type=int, default=300,
                        help="Image dpi")
    parser.add_argument("--num_of_plot_types",
                        type=int, default=2,
                        help="Numper of plots on a figure")
    parser.add_argument("--img_type",
                        type=str, default="plots",
                        help="Choose what to generate: plots or labels")
    parser.add_argument("--axis",
                        type=str, default="x",
                        help="if labels then choose which axis: x or y")
    # train
    parser.add_argument("--unet_depth",
                        type=int, default=3,
                        help="Depth of U-Net (encoder)")
    parser.add_argument("--depth",
                        type=int, default=3,
                        help="Depth of U-Net (encoder)")
    parser.add_argument("--device",
                        type=str, default="cpu",
                        help="cpu, gpu, mps")
    parser.add_argument("--batch_size",
                        type=int, default=32,
                        help="batch size")
    parser.add_argument("--epochs",
                        type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--output_freq",
                        type=int, default=2,
                        help="output frequency")
    parser.add_argument("--run_description",
                        type=str, default=None,
                        help="Name of the file with mlflow experiment description")
    parser.add_argument("--experiment_name",
                        type=str, default="Experiments",
                        help="Name of experiments, used to compares several runs")
    parser.add_argument("--lr",
                        type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weights",
                        type=str, default=None,
                        help="Weights of a pretrianed model")
    parser.add_argument("--dice_coef",
                        type=float, default=0.1,
                        help="Dice Loss contribution")
    # extract
    parser.add_argument("--my_img",
                        type=str, default="./plot_image.png",
                        help="pathe to an image with a plot")
    args = parser.parse_args()
    return args


def count_torch_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    trainable_weights = 0
    non_trainable_weights = 0

    for param in model.parameters():
        param_count = param.numel()
        if param.requires_grad:
            trainable_params += param_count
            trainable_weights += param_count * param.element_size()
        else:
            non_trainable_params += param_count
            non_trainable_weights += param_count * param.element_size()

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Trainable weights (Mb): {trainable_weights / 1e6}")
    print(f"Non-trainable weights (Mb): {non_trainable_weights / 1e6}")


def read_run_description(file_name: str = "description.txt"):
    with open(file_name, "r") as f:
        text = f.read()
    f.close()
    return text


def load_model(model: torch.nn.Module, file_name: str):
    """loads weights to the model"""
    assert os.path.exists(file_name), "No weights were found"
    state_dict = torch.load(file_name, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Models is loaded from: ", file_name)
    return model


def embedded_to_number(embedded: torch.tensor):
    """
    extracts numbers from an embedded prediciton
    """
    em = torch.argmax(embedded, axis=2)
    em = em[0].numpy()
    nums = []
    for i in range(2):
        sign = -1 + 2 * em[i * 3]
        num = int(f"{em[i*3 + 1]}{em[i*3 + 2]}")
        nums.append(sign * num)
    return nums


class Rescaler:
    # TODO actually I get positions of labels and not of the points
    """Each point of a segmented plot is expressed in pixels.
       The class estimates rescaling parameters
        to turn croodintaes from pixesl to an image coordinates.
        """
    def __init__(self, labels: dict, nums: dict) -> None:
        self.r0, self.delta = self.scaling_factor(labels, nums)

    def scaling_factor(self, labels, nums):
        """Finds (0,0) point (r0) in pixels
            as well as unit vector (delta) in pixels"""
        r0 = np.array([labels["x"][-1][0] + labels["x"][0][0],
                       labels["y"][-1][1] + labels["y"][0][1]]) / 2
        delta = np.array([(labels["x"][-1][0] - labels["x"][0][0]) / (nums["x"][1] - nums["x"][0]),
                          (labels["y"][0][1] - labels["y"][-1][1]) / (nums["y"][1] - nums["y"][0])])
        return r0, delta

    def rescale(self, point):
        """finds coordinates of a point given in pixels
            in image coordinates"""
        return (point - self.r0) / self.delta
