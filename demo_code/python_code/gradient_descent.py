# gradient_descent
import numpy as np

# get gradient function
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad
# gradient_descent
def gradient_descent(f, x, lr=0.01, step_num=100):
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

