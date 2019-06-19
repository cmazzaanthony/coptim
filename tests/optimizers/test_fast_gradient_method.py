import unittest

import numpy as np

from src.functions.quadratic import Quadratic
from src.optimizers.fast_gradient_method import FastGradientMethod


class TestFastGradientMethod(unittest.TestCase):

    def test_fast_gradient_method_with_quad_objective(self):
        objective = Quadratic()
        starting_point = np.array([1, 2, 3, 4])
        deltas = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        epsilon = 0.00001
        estimates = list()
        iterations = list()
        for delta in deltas:
            optimizer = FastGradientMethod()
            x = optimizer.optimize(starting_point,
                                   objective,
                                   epsilon,
                                   delta)

            estimates.append(x)
            iterations.append(optimizer.iterations)

        self.assertListEqual(iterations, [
            158,
            711,
            2122,
            3820,
            9469,
            9229,
        ])
