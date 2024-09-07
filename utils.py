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
    parser.add_argument("--framework", type=str, default="tensorflow", help="Choose between tensorflow and torch")
    parser.add_argument("--fig_size", type=int, default=5, help="Plots size")
    parser.add_argument("--depth", type=int, default=3, help="Depth of U-Net (encoder)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, gpu, mps")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--output_freq", type=int, default=2, help="output frequency")
    parser.add_argument("--save_history", type=str, default=None, help="name of the file to save history")
    parser.add_argument("--run_description", type=str, default=None, help="Name of the file with mlflow experiment description")
    parser.add_argument("--experiment_name", type=str, default="Experiments", help="Name of agglomirated experiments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weights", type=str, default=None, help="Weights of a pretrianed model")
    parser.add_argument("--dice_coef", type=float, default=0.9, help="Dice Loss contribution")
    parser.add_argument("--num_of_classes", type=int, default=2, help="Numper of plots on a figure")
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
    assert os.path.exists(file_name), "No weights were found"
    state_dict = torch.load(file_name, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Models is loaded from: ", file_name)
    return model
