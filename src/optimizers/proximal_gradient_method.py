import numpy as np

from src.optimizer import Optimizer


class ProximalGradientMethod(Optimizer):
    def __init__(self):
        self.iterations = 0

    def optimize(self, x_0, func, nsfunc, step_size):
        x = x_0
        for iterations in range(100):

            x = nsfunc.prox(x - step_size * func.gradient(x))

        return x

    def stopping_criteria(self, x_t, x, func, epsilon):
        return np.linalg.norm(x_t - x) <= epsilon

    # def step_size(self, x, func, beta, d, sigma):
    #     i = 0
    #     step = 1.0
    #     inequality_satisfied = True
    #     while orig_func <= quad_func:
    #         orig_func = sfunc.eval(x, beta_prox, y)
    #         quad_func = mse_val + delta @ mse_grad_val.flatten() + 0.5 * step * (delta @ delta)
    #
    #         step *= epsilon
    #
    #
    #     return np.power(beta, i)
