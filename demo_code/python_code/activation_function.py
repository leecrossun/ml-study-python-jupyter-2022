# Activation function
import numpy as np
import matplotlib.pylab as plt

# activation_function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# ouput-layer function
def identity_function(x):
    return x

def softmax(a):
    a -= np.max(a) # prevent overflow
    return np.exp(a) / np.sum(np.exp(a))

'''
# test code
x = np.arange(-5.0, 5.0, 0.1) # x축
sigmoid_y = sigmoid(x) # y축


plt.plot(x, sigmoid_y)
plt.ylim(-0.1, 1.1)
plt.show()

relu_y = relu(x)
plt.plot(x, relu_y)
plt.show()
'''