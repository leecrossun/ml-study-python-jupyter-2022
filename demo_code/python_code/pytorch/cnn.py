# CNN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torch.nn.init

# device set
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# parameter
learning_rate = 0.1
training_epochs = 15
batch_size = 1

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

# data load
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1), # apply convolution
            nn.ReLU(),
            nn.MaxPool2d(2)  # max pooling
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # apply convolution
            nn.ReLU(),
            nn.MaxPool2d(2)  # max pooling
        )

        self.fc = nn.Linear(7*7*64, 10, bias=True) # apply linear transformation (y = xA^T + b)
        torch.nn.init.xavier_uniform(self.fc.weight) # init weight
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device) # cost + softmax
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # 매개변수 update (cf. SGD)

total_batch = len(data_loader)
print('총 배치의 수 : {0} '.format(total_batch))
print(data_loader)

# main
avg_cost_list = []
for epoch in range(1, training_epochs + 1):
    avg_cost = 0

    for X, Y in data_loader: # mini batch 단위
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X) # 예측
        cost = criterion(hypothesis, Y) # cost
        cost.backward() # cost의 W에 대한 기울기 (미분)
        optimizer.step() # update

        avg_cost += cost / total_batch # 평균 cost
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch, avg_cost))

print('--training is finished--')
# accuracy test
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())

# x = len(avg_cost_list)
# y = avg_cost_list
# plt.plot(x, y)
# plt.show()
