import os
import sys
import unittest

REPO_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, REPO_DIR)

import numpy as np

class TestCase(unittest.TestCase):
    def assertArrayEqual(self, a, b, *args, **kwargs):
        np.testing.assert_array_equal(a, b, *args, **kwargs)

    def assertClose(self, a, b, *args, **kwargs):
        np.testing.assert_allclose(a, b, *args, **kwargs)
