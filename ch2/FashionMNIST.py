from torch import torch, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


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


class FashionNetwork:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # 데이터셋 로드
        transform = transforms.Compose([transforms.ToTensor()])

        self.train_dataset, self.test_dataset = (
            datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            ),
            datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform
            ),
        )

    # 정확도 계산 함수
    def get_accuracy(self, pred, labels):
        _, predictions = torch.max(pred, 1)
        correct = (predictions == labels).float().sum()
        accuracy = correct / labels.shape[0]
        return accuracy

    # 모델 훈련 함수
    def train(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        # 추후 정확도 계산을 위한 손실 및 정확도 총합
        total_loss, total_accuracy = 0, 0

        # 신경망 실행 및 파라미터 산출
        self.model.train()

        # 실행 결과를 기반으로 학습: 100개씩 나누어 배치 학습
        for batch, (X, y) in enumerate(dataloader):
            # 예측과 손실 계산
            pred = self.model(X)  # 신경망 파라미터
            loss = self.loss_fn(pred, y)  # 파라미터의 손실값
            accuracy = self.get_accuracy(pred, y)  # 정확도 산출 (그래디언트 아님)

            # 역전파
            self.optimizer.zero_grad()  # 옵티마이저 초기화
            loss.backward()  # 역전파 값 구하고 저장
            self.optimizer.step()  # 역전파 값을 기반으로 최적화 진행

            # 한 손실과 정확도를 계산할 때 마다 합해준다.
            total_loss += loss.item()
            total_accuracy += accuracy.item()

            # 배치 하나를 돌 때 마다 그 배치에서의 평균 정확도를 평가
            if batch % 100 == 0:
                current = batch * len(X)
                avg_loss = total_loss / (batch + 1)
                avg_accuracy = total_accuracy / (batch + 1) * 100
                print(
                    f"배치 {batch}, 손실: {avg_loss:>7f}, 정확도: {avg_accuracy:>0.2f}% [{current:>5d}/{size:>5d}]"
                )

            # 평균 정확도가 95가 넘는다면, 오버피팅을 방지하기 위해 훈련을 중단한다.
            if avg_accuracy >= 95:
                print("95% 정확도에 도달했으므로 훈련을 중지합니다.")
                return True

    # 테스트 함수
    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        # 모델을 추론 모드로 전환
        self.model.eval()

        # 테스트 데이터로 추론 실행
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # 틀린 값과 맞은 값으로 정확도와 평균 손실을 구한다.
        test_loss /= num_batches
        correct /= size
        print(f"테스트 오차:\n정확도:{(100*correct):>0.1f}%, 평균손실:{test_loss:>8f}")

    # 이미지 예측
    def predict_single_image(self, image, label):
        # 모델을 추론 모드로 전환
        self.model.eval()

        # unsqueeze가 뭐지?
        image = image.unsqueeze(0)

        # 이미지 데이터로 추론 실행
        with torch.no_grad():
            prediction = self.model(image)
            print(image)
            predicted_label = prediction.argmax(1).item()

        # 실제 이미지 데이터 그리기
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Predicted: {predicted_label}, Actual: {label}")
        plt.show()

        # 레이블 예측값
        return predicted_label


# 모델, 손실함수, 옵티마이저 정의
model = FashionMNISTModel()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# 네트워크 초기화
net = FashionNetwork(model, loss_function, optimizer)

# 데이터셋 가져오기
train_loader, test_loader = (
    DataLoader(net.train_dataset, batch_size=64, shuffle=True),
    DataLoader(net.test_dataset, batch_size=64, shuffle=False),
)

# 훈련 실행
epochs = 5
for t in range(epochs):
    print(f"에폭 {t+1}\n----------------------------")
    net.train(train_loader)
print("완료!")

# 모델 평가
net.test(test_loader)

# 모델 예측
image, label = net.test_dataset[0]
predicted_label = net.predict_single_image(image, label)
print(f"모델의 예측 {predicted_label}, 실제 레이블: {label}")
