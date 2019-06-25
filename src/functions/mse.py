from src.function import Function


class MSE(Function):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def eval(self, beta):
        diff = self.X @ beta - self.y
        return (1 / len(self.y)) * 0.5 * diff.T @ diff

    def gradient(self, beta):
        return (1 / len(self.y)) * -self.X.T @ (self.y - self.X @ beta)

    def hessian(self, *arg, **kwargs):
        pass
