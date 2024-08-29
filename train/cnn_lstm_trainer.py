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
                 num_classes: int,
                 device: str = "cpu:0",
                 **kwargs
                 ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.mlf = kwargs
        self.num_classes = num_classes
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(self.mlf["experiment_name"])

    def train_step(self,
                   dataloader: torch.utils.data.DataLoader
                   ):
        self.model.train()
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred.view(-1, self.num_classes), y.view(-1))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=2)
            correct_predictions += (y_pred == y).sum().item()
            total_predictions += y.numel()
        avg_train_loss = train_loss / len(dataloader)
        avg_train_accuracy = correct_predictions / total_predictions
        return avg_train_loss, avg_train_accuracy

    def validationt_step(self,
                         dataloader: torch.utils.data.DataLoader
                         ):
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.inference_mode():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred.view(-1, self.num_classes), y.view(-1))
                test_loss += loss.item()
                y_pred = torch.argmax(y_pred, dim=2)
                correct_predictions += (y_pred == y).sum().item()
                total_predictions += y.numel()
            avg_train_loss = test_loss / len(dataloader)
            avg_train_accuracy = correct_predictions / total_predictions
        return avg_train_loss, avg_train_accuracy

    def test_step(self,
                  dataloader: torch.utils.data.DataLoader,
                  epoch: int
                  ):
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.inference_mode():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred.view(-1, self.num_classes), y.view(-1))
                test_loss += loss.item()
                y_pred = torch.argmax(y_pred, dim=2)
                correct_predictions += (y_pred == y).sum().item()
                total_predictions += y.numel()
            avg_test_loss = test_loss / len(dataloader)
            avg_test_accuracy = correct_predictions / total_predictions
        print("Loss per test: ", avg_test_loss, "| Accuracy per test: ", avg_test_accuracy)
        y_pred = [tuple(digits.tolist()) for digits in y_pred]
        _, ax = plt.subplots(1, 5)
        X = X.to("cpu")
        for n in range(5):
            ax[n].imshow(X[n][0])
            ax[n].set_title(f"{y_pred[n][0]} {y_pred[n][1]}")
            ax[n].axis("off")
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
                train_loss, train_acc = self.train_step(train_dataloder)
                test_loss, test_acc = self.validationt_step(validation_dataloder)
                avg_train_loss[first_step + epoch] = train_loss
                avg_seg_loss[first_step + epoch] = test_loss
                if epoch == 1:
                    mlflow.log_artifact("./data/data_numbers.py")
                    mlflow.log_artifact("./models_zoo/cnn_lstm.py")
                if (epoch + 1) % 20 == 0:
                    saved_under = "./intermediate.pth"
                    torch.save(self.model.state_dict(), saved_under)
                    mlflow.log_artifact(saved_under)
                if (epoch + 1) % 5 == 0:
                    self.test_step(test_dataloder, epoch)
                if epoch % output_freq == 0:
                    print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                    Train loss: {train_loss:.5f} acc: {train_acc:.5f}")
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("train_acc", train_acc, step=epoch)
                    mlflow.log_metric("val_acc", test_acc, step=epoch)
        saved_under = f"./{self.mlf['run_name']}.pth"
        torch.save(self.model.state_dict(), saved_under)
        mlflow.log_artifact(saved_under)
        return avg_train_loss, avg_seg_loss
