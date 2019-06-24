import numpy as np

from src.optimizer import Optimizer


class ProximalGradientMethod(Optimizer):
    def __init__(self, design_matrix, response):
        self.design_matrix = design_matrix
        self.response = response
        self.iterations = 0

    def _backtracking_line_search(self, x, beta, y, weights, max_iter=20, epsilon=2.0):
        mse_val = self.sfunc.eval(x, beta, y)
        mse_grad_val = self.sfunc.gradient(x, beta, y)
        step = 1.0

        for ls in range(max_iter):
            beta_prox = self.nsfunc.prox(beta - mse_grad_val / step, weights / step)
            delta = (beta_prox - beta).flatten()

            orig_func = self.sfunc.eval(x, beta_prox, y)
            quad_func = mse_val + delta @ mse_grad_val.flatten() + 0.5 * step * (delta @ delta)

            if orig_func <= quad_func:
                break

            step *= epsilon

        return beta_prox

    def optimize(self, x_0, func, nsfunc, beta, sigma, epsilon):
        x = x_0
        for iterations in range(100):

            next_beta = self._backtracking_line_search(beta)

            if self.stopping_criteria(next_beta, beta):
                print(f'Threshold reached in {iterations}')
                break

            beta = next_beta

        return beta

    def stopping_criteria(self, x_t, x, func, epsilon):
        return np.linalg.norm(x_t - x) <= epsilon

    def step_size(self, x, func, beta, d, sigma):
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func.eval(x + np.power(beta, i) * d) <= func.eval(x) + np.power(beta, i) * sigma * func.gradient(x).dot(
                    d):
                break

            i += 1

        return np.power(beta, i)
