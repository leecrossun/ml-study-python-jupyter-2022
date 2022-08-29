# Relu class
class Relu:
    def __init__(self):
        self.mask = (none)

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # x 값 그대로
        out[self.mask] = 0 # x <= 0 이면 0

        return out

    def backward(self, x):
        dout[self.mask] = 0
        dx = dout

        return dx

