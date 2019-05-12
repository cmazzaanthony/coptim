import unittest

import numpy as np

from coptim.functions.quadratic import Quadratic
from coptim.optimizers.gradient_method_exact_minimization import GradientMethodExactMinimization


class TestGradientMethodExactMinimization(unittest.TestCase):

    def test_inexact_gradient_method(self):
        objective = Quadratic()
        starting_point = np.array([1, 2, 3, 4])
        deltas = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        epsilon = 0.00001
        estimates = list()
        iterations = list()
        for delta in deltas:
            optimizer = GradientMethodExactMinimization()
            x = optimizer.optimize(starting_point,
                                   objective,
                                   delta,
                                   epsilon)

            estimates.append(x)
            iterations.append(optimizer.iterations)

        self.assertListEqual(iterations, [
            23,
            91,
            761,
            7449,
            74321,
            743055
        ])
