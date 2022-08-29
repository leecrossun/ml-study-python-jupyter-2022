import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.functions import sigmoid, softmax, cross_entropy_error

class TwoLayerNet:
    # 가중치 초기화
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 신경망의 매개변수
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)}

        # 예측 실행
        def predict(self, x):
            W1, W2 = self.params['W1'], self.params['W2']
            b1, b2 = self.params['b1'], self.params['b2']

            a1 = np.dot(x, W1) + b1
            z1 = sigmoid(a1)

            a2 = np.dot(z1, W2) + b2
            y = softmax(a2)
            return y

        # 손실함수 (x : input, t : true label)
        def loss(self, x, t):
            y = self.predict(x)
            return cross_entropy_error(y, t)

        # 정확도
        def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            y = np.argmax(t, axis=2)

            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy

        # 가중치 매개변수의 기울기
        def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)

            # 기울기
            grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                     'b1': numerical_gradient(loss_W, self.params['b1']),
                     'W2': numerical_gradient(loss_W, self.params['W2']),
                     'b2': numerical_gradient(loss_W, self.params['b2'])}

            return grads
