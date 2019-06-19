import numpy as np

from src.optimizer import Optimizer


class FastGradientMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def optimize(self, x_0, func, epsilon, delta):
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, delta]])
        L = np.linalg.norm(Q)
        c = np.array([1, 1, 1, 1])
        x = x_0
        y = x_0
        alpha = 1
        while self.stopping_criteria(x, Q, c, func, epsilon):
            prev_x = x
            x = y - 1 / L * func.gradient(Q, c, y)

            prev_alpha = alpha
            alpha = (1 + np.sqrt(1 + 4 * np.power(prev_alpha, 2))) / 2

            y = x + (prev_alpha - 1) / alpha * (x - prev_x)

            self.iterations += 1

    def stopping_criteria(self, x, Q, c, func, epsilon):
        return np.linalg.norm(func.gradient(Q, c, x)) >= epsilon

    def step_size(self, x, func, beta, d, sigma):
        pass
