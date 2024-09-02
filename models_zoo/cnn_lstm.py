from torch import nn
from torch.functional import F


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_LSTM, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # output: (32, 30, 250)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: (32, 15, 125)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # output: (64, 15, 125)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # output: (64, 15, 125)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # output: (64, 7, 62)
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes * 6)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1, 64)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(-1, 6, self.num_classes)
        x = F.log_softmax(x, dim=2)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    num_classes = 10
    model = CNN_LSTM(num_classes=10)

    summary(model, input_size=(1, 1, 30, 250))
