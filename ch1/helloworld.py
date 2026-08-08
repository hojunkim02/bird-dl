import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 모델
model = nn.Sequential(nn.Linear(1, 1))  # 우리가 개선해나갈 모델

# 손실함수와 옵티마이저
criterion = nn.MSELoss()  # 손실함수를 지정
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 옵티마이저에 모델의 매개변수 주입

# 데이터
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

# 훈련
for _ in range(500):
    optimizer.zero_grad()  # 모델의 미분값 초기화 (옵티마이저가 모델 포함 중)
    outputs = model(xs)  # output(텐서)에 model 내부의 파라미터 전송
    loss = criterion(outputs, ys)  # loss(텐서)에 output 내부의 파라미터 전송
    loss.backward()  # 모델의 w, b 값에 대한 미분값 저장 (모델에 저장됨)
    optimizer.step()  # w, b값의 미분값을 통해 최적화 실행

# 예측
with torch.no_grad():
    # with 문이란: __enter__, __exit__ 을 미리 정의하여 효율적으로 코드 작성
    print(model(torch.tensor([[10.0]], dtype=torch.float32)))

    # 모델의 첫 층을 얻는다
    layer = model[0]

    # 가중치와 편향을 추출한다
    weight = layer.weight.data.numpy()
    bias = layer.bias.data.numpy()
    print("Weight:", weight)
    print("Bias:", bias)
