import numpy as np

from src.function import Function


class L1(Function):

    def __init__(self, _lambda):
        self._lambda = _lambda

    def eval(self, beta):
        self._lambda * np.absolute(beta)

    def prox(self, beta):
        """
        soft-thresholding operator
        """
        return np.sign(beta) * np.maximum(np.abs(beta) - self._lambda, 0)

    def gradient(self, x, beta, y):
        pass

    def hessian(self, *arg, **kwargs):
        pass
