# 미니배치 학습
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# hyper parameter
iters_num = 10000
train_size = x_train.shape[0] # the number of rows
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    if i % 1000 == 0:
        print("{0} 번째 batch".format(i))

    # mini batch
    batch_mask = np.random.choice(train_size, batch_size) # train_size에서 batch_size 만큼의 random sample 추출
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # get gradient
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) # 성능 개선

    # update param
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key] # lr(학습률) * 기울기 값 update


    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % 1000 == 0:
        print("loss value : {0}".format(train_loss_list[i]))

x = np.arange(0, len(train_loss_list))
y = train_loss_list
plt.plot(x, y)
plt.show()