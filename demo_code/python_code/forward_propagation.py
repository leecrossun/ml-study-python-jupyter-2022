# forward_propagation - 손글씨 숫자 인식하기
import numpy as np
from activation_function import relu, softmax
from forward import forward
import sys, os
from dataset.mnist import load_mnist
import pickle
sys.path.append(os.pardir)

# MNIST dataset load
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


# pickle file(sample_weight.pkl)에 저장된 '학습된 가중치 매개변수' load
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

x, t = get_data()
network = init_network()
accuracy_cnt = 0 # 정확도 판별

for i in range(len(x)):
    y = forward(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 index 얻기

    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: ", str(float(accuracy_cnt) / len(x)))
