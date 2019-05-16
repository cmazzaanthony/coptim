import numpy as np

from coptim.optimizer import Optimizer


class BFGSMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def psi(self, t, sigma, x, d, func, gradient):
        return func(x + t * d) - \
               func(x) - \
               sigma * t * gradient(x).dot(d)

    def wolfe_powell_rule_phase_B(self, a, b, rho, sigma, x, d, func):
        a_j = a
        b_j = b
        stop = False

        while not stop:
            t_j = a_j + (b_j - a_j) / 2

            if self.psi(t_j, sigma, x, d, func, gradient) >= 0:
                b_j = t_j

            if self.psi(t_j, sigma, x, d, func, gradient) < 0 and \
                    gradient(x + t_j * d).dot(d) >= rho * gradient(x).dot(d):
                t = t_j
                stop = True

            if self.psi(t_j, sigma, x, d, func, gradient) < 0 and \
                    gradient(x + t_j * d).dot(d) < rho * gradient(x).dot(d):
                a_j = t_j

        return t

    def wolfe_powell_rule_phase_A(self, rho, sigma, x, d, func):
        gamma = 2
        t_i = 1
        stop = False
        i = 0

        while not stop:
            if self.psi(t_i, sigma, x, d, func) >= 0:
                a = 0
                b = t_i

                t_i = self.wolfe_powell_rule_phase_B(a, b, rho, sigma, x, d, func)

            if self.psi(t_i, sigma, x, d, func, gradient) < 0 and \
                    gradient(x + t_i * d).dot(d) >= rho * gradient(x).dot(d):
                t = t_i
                stop = True

            if self.psi(t_i, sigma, x, d, func, gradient) < 0 and \
                    gradient(x + t_i * d).dot(d) < rho * gradient(x).dot(d):
                i += 1
                t_i = gamma * t_i

        return t

    def optimize(x_0, H_0, rho, sigma, epsilon, func, gradient):
        x = x_0
        H = H_0
        k = 0
        while np.linalg.norm(gradient(x)) > epsilon:
            nabla_x = -gradient(x)

            inverse = np.linalg.inv(H)
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
