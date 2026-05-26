import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 모델 정의
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = FashionMNISTModel()


# 손실함수와 옵티마이저 정의
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


# 정확도 계산 함수
def get_accuracy(pred, labels):
    _, predictions = torch.max(pred, 1)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy


# 모델 훈련 함수
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, total_accuracy = 0, 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 예측과 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = get_accuracy(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if batch % 100 == 0:
            current = batch * len(X)
            avg_loss = total_loss / (batch + 1)
            avg_accuracy = total_accuracy / (batch + 1) * 100
            print(
                f"배치 {batch}, 손실: {avg_loss:>7f}, 정확도: {avg_accuracy:>0.2f}% [{current:>5d}/{size:>5d}]"
            )

        if avg_accuracy >= 95:
            print("95% 정확도에 도달했으므로 훈련을 중지합니다.")
            return True


# 훈련 실행
epochs = 5
for t in range(epochs):
    print(f"에폭 {t+1}\n----------------------------")
    train(train_loader, model, loss_function, optimizer)
print("완료!")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"테스트 오차:\n 정확도:{(100*correct):>0.1f}%, 평균손실:{test_loss:>8f}\n")


# 모델 평가
test(test_loader, model)


def predict_single_image(image, label, model):
    model.eval()
    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
        print(image)
        predicted_label = prediction.argmax(1).item()

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Actual: {label}")
    plt.show()

    return predicted_label


image, label = test_dataset[0]
predicted_label = predict_single_image(image, label, model)
print(f"모델의 예측 {predicted_label}, 실제 레이블: {label}")
