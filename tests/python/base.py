import os
import sys
import unittest

REPO_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, REPO_DIR)

class TestCase(unittest.TestCase):
    pass
