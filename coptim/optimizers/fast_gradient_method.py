from coptim.optimizer import Optimizer


class FastGradientMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def optimize(self, x_0, func, beta, sigma, epsilon):
        pass

    def stopping_criteria(self, x, func, epsilon):
        pass

    def step_size(self, x, func, beta, d, sigma):
        pass
