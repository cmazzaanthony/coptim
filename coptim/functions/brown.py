import numpy as np
from coptim.function import Function


class Brown(Function):

    def eval(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        return (np.power(x[0] - 1e6, 2) +
                np.power((x[1] - (2 * 1e-6)), 2) +
                np.power((x[0] * x[1] - 2), 2))

    def gradient(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        return np.array([
            2 * x[0] * (x[1] ** 2 + 1) - 4 * (x[1] + 500000),
            2 * (x[0] * (x[0] * x[1] - 2) + x[1] - (1 / 500000))
        ])

    def hessian(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        dfdx1 = 2 + (2 * x[1] ** 2)
        dfdx1dx2 = 4 * x[0] * x[1] - 4
        dfdx2dx1 = 4 * x[0] * x[1] - 4
        dfdx2 = 2 + (2 * x[0] ** 2)

        return np.array([[dfdx1, dfdx1dx2], [dfdx2dx1, dfdx2]])
