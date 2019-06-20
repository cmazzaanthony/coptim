import numpy as np
from src.function import Function


def func_sum(x):
    total = 0
    for i in range(1, len(x) + 1):
        total += i * (x[i - 1] - 1)

    return total


class Func1(Function):

    def eval(self, x):
        total = 0
        for i in range(len(x)):
            Fi = x[i] - 1
            total += Fi ** 2

        return total + np.power(func_sum(x), 2) + np.power(func_sum(x), 4)

    def gradient(self, x):
        fs = func_sum(x)
        grad = 2 * (x - 1)
        for i in range(1, len(x) + 1):
            grad[i - 1] = grad[i - 1] + 2 * i * fs + 4 * i * np.power(fs, 3)

        return grad

    def hessian(self, x):
        fs = func_sum(x)
        hess_mat = np.zeros(shape=(len(x), len(x)))

        for i in range(1, len(x) + 1):
            for j in range(1, len(x) + 1):
                if i == j:
                    hess_mat[i - 1, j - 1] = 2 + 2 * i * j + 12 * i * j * np.power(fs, 2)

                else:
                    hess_mat[i - 1, j - 1] = 2 * i * j + 12 * i * j * np.power(fs, 2)

        return hess_mat
