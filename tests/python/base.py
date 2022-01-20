import os
import sys
import unittest

REPO_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, REPO_DIR)

import cubismup2d as cup2d

import numpy as np

class TestSimulation(cup2d.Simulation):
    def __init__(self, *args, argv=[], **kwargs):
        super().__init__(*args, argv=['-maxPoissonIterations', '10'] + argv, **kwargs)


class TestCase(unittest.TestCase):
    def assertArrayEqual(self, a, b, *args, **kwargs):
        np.testing.assert_array_equal(a, b, *args, **kwargs)

    def assertClose(self, a, b, *args, **kwargs):
        np.testing.assert_allclose(a, b, *args, **kwargs)

    def assertArrayAlmostEqual(self, a, b, *args,
                               check_shape: bool = False, **kwargs):
        """
        Arguments:
            check_shape: if set, asserts that shapes are equal
        """
        if check_shape:
            self.assertEqual(a.shape, b.shape)
        np.testing.assert_almost_equal(a, b, *args, **kwargs)
