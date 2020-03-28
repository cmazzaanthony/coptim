import numpy as np

from coptim.optimizer import Optimizer


class ProximalGradientMethod(Optimizer):
    def __init__(self):
        self.iterations = 0
        self.max_iters = 100

    def optimize(self, x_0, func, nsfunc, step_size, epsilon):
        x = x_0
        for iterations in range(self.max_iters):

            next_x = nsfunc.prox(x - step_size * func.gradient(x))

            if self.stopping_criteria(next_x, x, epsilon):
                break

            x = next_x
            self.iterations += 1

        return x

    def stopping_criteria(self, x_t, x, epsilon):
        return np.linalg.norm(x_t - x) <= epsilon
