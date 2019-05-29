import numpy as np
from coptim.function import Function


class Bateman(Function):

    def eval(self, x):
        y_x_t = x[2] * (np.exp(-x[0] * t_list) - np.exp(-x[1] * t_list))
        f_raw = np.power(y_x_t - Y, 2)

        return 0.5 * np.sum(f_raw)

    def gradient(self, x):
        y_x_t = x[2] * (np.exp(-x[0] * T) - np.exp(-x[1] * T))

        dx1 = 0
        dx2 = 0
        dx3 = 0
        for i in range(13):
            dx1 = dx1 + (x[2] * -T[i] * (np.exp(-x[0] * T[i]))) * (y_x_t[i] - Y[i])
            dx2 = dx2 + (x[2] * T[i] * np.exp(-x[1] * T[i])) * (y_x_t[i] - Y[i])
            dx3 = dx3 + (np.exp(-x[0] * T[i]) - np.exp(-x[1] * T[i])) * (y_x_t[i] - Y[i])

        return np.array([dx1, dx2, dx3])


    def hessian(self, x):
        pass
