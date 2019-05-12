from coptim.optimizer import Optimizer


class FastGradientMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def optimize(self, x_0, func, beta, sigma, epsilon):
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, delta]])
        L = np.linalg.norm(Q)
        c = np.array([1, 1, 1, 1])
        x = x_0
        y = x_0
        alpha = 1
        k = 0
        while np.linalg.norm(dfunct(Q, c, x)) >= epsilon:
            prev_x = x
            x = y - 1 / L * dfunct(Q, c, y)

            prev_alpha = alpha
            alpha = (1 + np.sqrt(1 + 4 * np.power(prev_alpha, 2))) / 2

            y = x + (prev_alpha - 1) / alpha * (x - prev_x)

            k += 1

    def stopping_criteria(self, x, func, epsilon):
        return np.linalg.norm(dfunct(Q, c, x)) >= epsilon

    def step_size(self, x, func, beta, d, sigma):
        pass
