# gradient_descent_main
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규 분포로 초기화

    # 예측 수행 함수
    def predict(self, x):
        return np.dot(x, self.W)

    # loss 값
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

# main
net = simpleNet()
print('net : \n',net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print('p : \n', p)

np.argmax(p)

t = np.array([0, 0, 1])
net.loss(x, t)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print('dW : \n', dW)