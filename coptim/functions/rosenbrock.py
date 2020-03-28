import numpy as np
from coptim.function import Function


class Rosenbrock(Function):

    def eval(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def gradient(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        return np.array([
            2 * (-200 * x[0] * x[1] + 200 * np.power(x[0], 3) - 1 + x[0]),
            200 * (x[1] - x[0] ** 2)
        ])

    def hessian(self, x):
        assert len(x) == 2, '2 dimensional input only.'
        df_dx1 = -400 * x[1] + 1200 * x[0] ** 2 + 2
        df_dx1dx2 = -400 * x[0]
        df_dx2dx1 = -400 * x[0]
        df_dx2 = 200

        return np.array([[df_dx1, df_dx1dx2], [df_dx2dx1, df_dx2]])
