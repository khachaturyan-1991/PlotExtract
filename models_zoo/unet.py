from torch.functional import F
from torch import nn
from torchinfo import summary


CONV_KERNEL = (3, 3)
POOL_KERNEL = (2, 2)
STRIDE_SIZE = (1, 1)
PADDING_TYPE = 1
MODEL_DEPTH = 3


class DownBlock(nn.Module):
    "Conv -> Batch -> ReLu -> Conv -> Batch -> ReLu -> MaxPool"
    def __init__(self,
                 in_channels,
                 out_channels, **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=CONV_KERNEL,
                                padding=PADDING_TYPE)
        self.batch_1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=CONV_KERNEL,
                                padding=PADDING_TYPE)
        self.batch_2 = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=POOL_KERNEL)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class UpperBlock(nn.Module):
    "ConvTranspose -> Conv -> interpolate"
    def __init__(self,
                 in_channels: int,
                 out_channels: int, **kwargs):
        super(UpperBlock, self).__init__(**kwargs)
        self.convT = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=CONV_KERNEL,
                                        stride=STRIDE_SIZE,
                                        padding=PADDING_TYPE)
        self.batch_T = nn.BatchNorm2d(num_features=out_channels)
        self.conv_1 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=CONV_KERNEL,
                                stride=STRIDE_SIZE,
                                padding=PADDING_TYPE)
        self.batch_1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, w=None):
        x = self.convT(x)
        x = self.batch_T(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu(x)
        s = x.shape[2:]
        x = F.interpolate(x,
                          size=(s[0] * 2, s[1] * 2),
                          mode='bilinear',
                          align_corners=False)
        return x


class UNet(nn.Module):

    def __init__(self, num_of_classes: int = 2, depth: int = MODEL_DEPTH, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.noc = num_of_classes
        self.depth = depth
        # encoder
        self.down_blocks = nn.ModuleDict()
        self.down_blocks[str(0)] = DownBlock(3, 64)
        n_feat = 64
        for i in range(1, depth):
            self.down_blocks[str(i)] = DownBlock(n_feat, 2 * n_feat)
            n_feat *= 2
        # decoder
        self.up_block = nn.ModuleDict()
        n_feat = 64 * (2 ** (depth - 1)) // 2
        for i in range(1, depth):
            self.up_block[str(depth - i)] = UpperBlock(2 * n_feat, n_feat)
            n_feat //= 2
        self.up_block[str(0)] = UpperBlock(64, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down_blocks[str(0)](x)
        for i in range(1, self.depth):
            x = self.down_blocks[str(i)](x)
        for i in range(1, self.depth):
            x = self.up_block[str(self.depth - i)](x)
        x = self.up_block[str(0)](x)
        x[:, 0, :, :] = self.sigmoid(x[:, 0, :, :])
        x[:, 1, :, :] = self.sigmoid(x[:, 1, :, :]) * self.noc
        return x


if __name__ == "__main__":
    import torch
    model = UNet(depth=3)
    summary(model, input_size=(1, 3, 128, 128))

    X = torch.randn(1, 3, 128, 128)
    result = model(X)
    print("Output shape: ", result.shape)
