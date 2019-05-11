import unittest

import numpy as np

from coptim.functions.brown import Brown
from coptim.optimizers.gradient_method_exact_minimization import GradientMethodExactMinimization


class TestGradientMethodExactMinimization(unittest.TestCase):

    def test_inexact_gradient_method(self):
        deltas = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        for delta in deltas:
            gradient_method(x_0=np.array([1, 2, 3, 4]),
                            func=Brown(),
                            delta=delta,
                            epsilon=0.00001)

            objective = Brown()
            starting_point = np.array([-1.2, 1])
            beta = 0.5
            sigma = 0.0001
            epsilon = 0.0001

            optimizer = GradientMethodExactMini/c/mization()

            x = optimizer.optimize(starting_point,
                                   objective,
                                   beta,
                                   sigma,
                                   epsilon)

