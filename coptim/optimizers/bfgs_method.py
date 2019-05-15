import numpy as np

from coptim.optimizer import Optimizer


class BFGSMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def optimize(x_0, H_0, rho, sigma, epsilon, func, gradient):
        x = x_0
        H = H_0
        k = 0
        while np.linalg.norm(gradient(x)) > epsilon:
            nabla_x = -gradient(x)

            inverse = inv(H)
            d = inverse.dot(nabla_x)

            step_size = wolfe_powell_rule_phase_A(rho=rho,
                                                  sigma=sigma,
                                                  x=x,
                                                  d=d,
                                                  func=func,
                                                  gradient=gradient)

            # print("step size: {}".format(step_size))
            # print("gradient {}".format(nabla_x))
            x_new = x + step_size * d
            s = x_new - x
            y = gradient(x_new) - gradient(x)

            H = H + (np.outer(y, y.T) / np.dot(y.T, s)) - np.outer(H.dot(s), s.T).dot(H) / (np.dot(np.dot(s.T, H), s))

            # print("new H: {}".format(H))
            # print("new x: {}".format(x_new))
            x = x_new
            k += 1

        print("final x: {}".format(x_new))
        print("iterations needed {}".format(k))
        return k


def stopping_criteria(self, x, func, epsilon):
        return np.linalg.norm(func.gradient(x)) >= epsilon

    def step_size(self, x, func, beta, d, sigma):
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func.eval(x + np.power(beta, i) * d) <= func.eval(x) + np.power(beta, i) * sigma * func.gradient(x).dot(d):
                break

            i += 1

        return np.power(beta, i)
