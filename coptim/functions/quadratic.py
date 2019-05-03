import numpy as np
from coptim.function import Function


class Quadratic(Function):

    def eval(self, Q, c, x, gamma):
        return 0.5 * x.T.dot(Q).dot(x) + c + gamma
