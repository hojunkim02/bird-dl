import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        # 합성곱 계층 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 1x28x28 -> 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64x14x14
        )

        # 합성곱 계층 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),  # -> 64x12x12
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x6x6
        )

        # 완전연결 계층
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
