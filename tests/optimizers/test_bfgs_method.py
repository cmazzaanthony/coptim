import unittest

import numpy as np

from coptim.functions.quadratic import Quadratic


class TestBFGSMethod(unittest.TestCase):

    def test_bfgs_method_with_quad_objective(self):
        objective = Quadratic()
        starting_point = np.array([1, 2, 3, 4])
        deltas = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
        epsilon = 0.00001
        estimates = list()
        iterations = list()
        for delta in deltas: