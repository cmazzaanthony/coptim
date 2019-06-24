import numpy as np

from src.optimizer import Optimizer


class ProximalGradientMethod(Optimizer):
    def __init__(self, design_matrix, response):
        self.design_matrix = design_matrix
        self.response = response
        self.iterations = 0

    # def _backtracking_line_search(self, x, sfunc, beta, nsfunc, y, weights,  epsilon=2.0):
    #     mse_val = sfunc.eval(x, beta, y)
    #     mse_grad_val = sfunc.gradient(x, beta, y)
    #     step = 1.0
    #
    #     for ls in range(20):
    #         beta_prox = nsfunc.prox(beta - mse_grad_val / step, weights / step)
    #         delta = (beta_prox - beta).flatten()
    #
    #
    #         if orig_func <= quad_func:
    #             break
    #
    #         step *= epsilon
    #
    #     return beta_prox

    def optimize(self, x_0, func, nsfunc, epsilon, step_size):
        x = x_0
        for iterations in range(100):

            x = nsfunc.prox(x - step_size * func.gradient(self.X, self.y, x))

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
