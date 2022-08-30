# logistic_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# param
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 10000
cost_list = []
for epochs in range(1, nb_epochs+1):
    # Cost
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    #print(hypothesis[:5])
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost to H(x)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    cost_list.append(cost.item())
    # print Log
    if epochs % 1000 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%".format(epochs, nb_epochs, cost.item(), accuracy * 100))
        print(prediction[:5])
        print(y_train[:5])

x = torch.arange(0, len(cost_list))
plt.plot(x, cost_list)
plt.show()