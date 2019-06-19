import numpy as np
from src.function import Function


class Quadratic(Function):

    def eval(self, Q, c, x, gamma):
        return 0.5 * x.T.dot(Q).dot(x) + c + gamma

    def gradient(self, Q, c, x):
        return Q.dot(x) + c

    def hessian(self):
        pass
