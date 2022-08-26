# cost_function
import numpy as np


# mean_squared_error_function, MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# cross_entropy_error, CEE
def cross_entropy_error(y, t):
    delta = 1e-7  # -inf 방지
    return -np.sum(t * np.log(y + delta))


# cross_entropy_error_batch
def cross_entropy_error_batch(y, t, one_hot_encoding):
    if y.dim == 1:
        y, t = y.reshape(1, y.size), t.reshape(1, t.size)

    batch_size = y.shape[0]
    if one_hot_encoding:
        return -np.sum(t * np.log(y)) / batch_size
    elif not one_hot_encoding:
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


'''
# main
# 실제 정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# ex. '2'일 확률이 가장 높다고 추정 (0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('ex1 result : ', cross_entropy_error(np.array(y), np.array(t)))

# ex2. '7'일 확률이 가장 높다고 추정 (0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('ex2 result : ', cross_entropy_error(np.array(y), np.array(t)))
'''
