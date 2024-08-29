from torch import nn
from torch.functional import F


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=128,
                           num_layers=2,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), 49, 64)
        x, _ = self.rnn(x)
        x = x[:, -2:, :]
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    num_classes = 10
    model = CNN_LSTM(num_classes=10)

    input_image = torch.rand(1, 1, 30, 30)
    summary(model, input_size=(1, 1, 30, 30))
