# Affine
import torch
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = torch.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = torch.dot(dout, self.W.T)
        self.dW = torch.dot(self.x.T, dout)
        self.db = torch.sum(dout, axis=0)
        return dx
