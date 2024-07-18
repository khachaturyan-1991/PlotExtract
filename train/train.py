import torch
from torch import nn
import tqdm

import mlflow  # type: ignore
import mlflow.pytorch  # type: ignore

import matplotlib.pylab as plt


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim,
                 device: str = "cpu:0"
                 ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def log_weights_and_gradients(self, step):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                mlflow.log_metric(f"{name}_weight_mean", param.data.mean().item(), step=step)
                mlflow.log_metric(f"{name}_weight_std", param.data.std().item(), step=step)
                if param.grad is not None:
                    mlflow.log_metric(f"{name}_grad_mean", param.grad.mean().item(), step=step)
                    mlflow.log_metric(f"{name}_grad_std", param.grad.std().item(), step=step)

    def train_step(self,
                   dataloader: torch.utils.data.DataLoader
                   ):
        self.model.train()
        step_loss = 0
        n = 0
        for X, masks in dataloader:
            X = torch.tensor(X, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            X = X.to(self.device)
            masks = masks.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, masks)
            step_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            n += 1
        return step_loss / n

    def test_step(self,
                  dataloader: torch.utils.data.DataLoader,
                  ):
        self.model.eval()
        step_loss = 0
        n = 0
        for X, masks in dataloader:
            X = torch.tensor(X, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            X = X.to(self.device)
            masks = masks.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, masks)
            step_loss += loss.item()
            n += 1
        return step_loss / n

    def fit(self,
            train_dataloder: torch.utils.data.DataLoader,
            test_dataloder: torch.utils.data.DataLoader,
            output_freq: int = 2,
            first_step: int = 0,
            epochs: int = 10):
        with mlflow.start_run() as _:  # it was run
            last_step = first_step + epochs
            avg_train_loss = {}
            avg_val_loss = {}
            for epoch in tqdm.tqdm(range(epochs)):
                train_loss = self.train_step(train_dataloder)
                test_loss = self.test_step(test_dataloder)
                avg_train_loss[first_step + epoch] = train_loss
                avg_val_loss[first_step + epoch] = test_loss
                if epoch % output_freq == 0:
                    print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                        Train loss: {train_loss:.4f} Validation loss: {test_loss:.4f}")
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("validation_loss", test_loss, step=epoch)
            mlflow.pytorch.log_model(self.model, "models")
        return avg_train_loss, avg_val_loss


def test_prediction(model,
                    dataloader,
                    loss_fn,
                    image_name: str = "test_prediction"):
    model.to("cpu")
    loss = 0
    n = 0
    for X, masks in dataloader:
        X = torch.tensor(X, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        Y_pred = model(X)
        loss += loss_fn(Y_pred, masks).item()
        n += 1

    print("Loss per test: ", loss / n)

    Y_pred = torch.squeeze(Y_pred, dim=-1)
    Y_pred = Y_pred.detach().numpy()
    masks = masks.numpy()
    _, axes = plt.subplots(2, 4, figsize=(8, 5))
    axes = axes.ravel()
    for i in range(4):
        axes[i].imshow(Y_pred[i][0])
        axes[i].axis("off")
        axes[i + 4].imshow(masks[i][0])
        axes[i + 4].axis("off")
    plt.tight_layout()
    plt.savefig(f"{image_name}.png")
