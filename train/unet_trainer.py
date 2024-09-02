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
                 segment_loss: nn.Module,
                 numeric_loss: nn.Module,
                 optimizer: torch.optim,
                 device: str = "cpu:0",
                 **kwargs
                 ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.segment_loss = segment_loss
        self.numeric_loss = numeric_loss
        self.optimizer = optimizer
        self.mlf = kwargs
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(self.mlf["experiment_name"])

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
            self.optimizer.zero_grad()
            X = X.type(torch.float32)
            X = X.to(self.device)
            masks = masks.type(torch.float32)
            masks = masks.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, masks)
            step_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            n += 1
        return step_loss / n

    def validationt_step(self,
                         dataloader: torch.utils.data.DataLoader
                         ):
        self.model.eval()
        step_loss = 0
        step_accuracy = 0
        n = 0
        with torch.inference_mode():
            for X, masks in dataloader:
                X = X.type(torch.float32)
                X = X.to(self.device)
                masks = masks.type(torch.float32)
                masks = masks.to(self.device)
                pred = self.model(X)
                segment_loss = self.segment_loss(pred, masks)
                numeric_loss = self.numeric_loss(pred, masks.float())
                step_loss += segment_loss.item()
                step_accuracy += numeric_loss.item()
                n += 1
        return step_loss / n, step_accuracy / n

    def test_step(self,
                  dataloader: torch.utils.data.DataLoader,
                  epoch: int
                  ):
        self.model.eval()
        step_loss = 0
        n = 0
        with torch.inference_mode():
            for X, masks in dataloader:
                X = X.type(torch.float32)
                X = X.to(self.device)
                masks = masks.type(torch.float32)
                masks = masks.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, masks)
                step_loss += loss.item()
                n += 1
        print("Loss per test: ", step_loss / n)
        pred = pred.cpu()
        masks = masks.cpu()
        masks = masks.numpy()
        pred = torch.squeeze(pred, dim=-1)
        pred = pred.detach().numpy()
        _, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        for i in range(4):
            axes[i].imshow(pred[i][0])
            axes[i].axis("off")
            axes[i + 4].imshow(pred[i][1])
            axes[i + 4].axis("off")
            axes[i + 8].imshow(masks[i][0, :, :])
            axes[i + 8].axis("off")
            axes[i + 12].imshow(masks[i][1, :, :])
            axes[i + 12].axis("off")
        plt.tight_layout()
        plt.savefig(f"./res_imgs/{epoch}.png")

    def fit(self,
            train_dataloder: torch.utils.data.DataLoader,
            validation_dataloder: torch.utils.data.DataLoader,
            test_dataloder: torch.utils.data.DataLoader,
            output_freq: int = 2,
            first_step: int = 0,
            epochs: int = 10):
        last_step = first_step + epochs
        avg_train_loss = {}
        avg_seg_loss = {}
        with mlflow.start_run(run_name=self.mlf["run_name"]) as _:
            for key in self.mlf.keys():
                mlflow.log_param(key, self.mlf[key])
            mlflow.set_tag("mlflow.note.content", self.mlf["run_description"])
            mlflow.log_param("device", self.device)
            for epoch in tqdm.tqdm(range(epochs)):
                train_loss = self.train_step(train_dataloder)
                seg_loss, num_loss = self.validationt_step(validation_dataloder)
                avg_train_loss[first_step + epoch] = train_loss
                avg_seg_loss[first_step + epoch] = seg_loss
                if epoch == 1:
                    mlflow.log_artifact("./metrics/losses.py")
                    mlflow.log_artifact("./data/data_plots.py")
                    mlflow.log_artifact("./models_zoo/unet.py")
                if (epoch + 1) % 20 == 0:
                    saved_under = "./intermediate.pth"
                    torch.save(self.model.state_dict(), saved_under)
                    mlflow.log_artifact(saved_under)
                if (epoch + 1) % 5 == 0:
                    self.test_step(test_dataloder, epoch)
                if epoch % output_freq == 0:
                    print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                    Train loss: {train_loss:.5f} Dice: {seg_loss:.5f} MSE: {num_loss:.5f}")
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("Dice", seg_loss, step=epoch)
                    mlflow.log_metric("MSE", num_loss, step=epoch)
        saved_under = f"./{self.mlf['run_name']}.pth"
        torch.save(self.model.state_dict(), saved_under)
        mlflow.log_artifact(saved_under)
        return avg_train_loss, avg_seg_loss
