# softmax_with_loss
from common.functions import cross_entropy_error, softmax
class softmax_with_loss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    def backward(self, x, t):
        batch_size = self.t.shape[0]
