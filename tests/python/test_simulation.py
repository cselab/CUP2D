from base import TestCase
import cubismup2d as cup2d

import numpy as np

class TestSimulation(TestCase):
    def test_field_load_uniform(self):
        sim = cup2d.Simulation(cells=(128, 64), nlevels=1)
        sim.init()
        tmp = np.random.uniform(0.0, 1.0, (64, 128))  # (y, x)
        sim.fields.chi.load_uniform(tmp)
        self.assertArrayEqual(tmp, sim.fields.chi.to_uniform())

        with self.assertRaises(TypeError):
            sim.fields.chi.load_uniform(tmp.T)
