# linear_regression (pytorch)
import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

# Data 정의, Hypothesis 초기화, Optimizer 정의 (최초 1회)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [4], [5]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer (gradient 를 구해서 weight 값을 변화시킴)
# SGD (Stochastic Gradient Descent)
optimizer = optim.SGD([W, b], lr=0.01) #  경사하강법

# Hypothesis 예측, Cost 계산, Optimizer로 학습 (반복)
nb_epochs = 10000
cost_list = []
for epochs in range(1, nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)  # 비용

    # Gradient Descent
    optimizer.zero_grad()  # init gradient
    cost.backward()  # gradient 계산 (역전파)
    optimizer.step()  # gradient 개선

    if epochs % 100 == 0:
        print("{0} 번째 cost : {1}".format(epochs, cost))
    cost_list.append(cost.tolist())

# draw graph
x = np.arange(0, len(cost_list))
y = cost_list
plt.plot(x, y)
plt.show()

