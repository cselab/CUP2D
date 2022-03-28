from base import TestCase, TestSimulation, cup2d

import numpy as np

import os

class TestShapes(TestCase):
    def _compute_moments(self, c: np.ndarray, dx: float, dy: float,
                         x: float, y: float):
        relx = (0.5 + np.arange(c.shape[1])) * dx - x
        rely = (0.5 + np.arange(c.shape[0])) * dy - y
        relx = relx[None, :]
        rely = rely[:, None]
        dA = dx * dy
        return {
            '0': c.sum() * dA,
            '1x': (c * relx).sum() * dA,
            '1y': (c * rely).sum() * dA,
            '2xx': (c * relx * relx).sum() * dA,
            '2yy': (c * rely * rely).sum() * dA,
            '2xy': (c * relx * rely).sum() * dA,
        }

    def test_disk(self):
        sim = TestSimulation(cells=(64, 32), nlevels=4, start_level=1, extent=100.0)
        disk = cup2d.Disk(sim, r=5.0, center=(40.0, 20.0))
        sim.add_shape(disk)
        sim.init()
        sim.simulate(nsteps=1)
        c = sim.fields.chi.to_uniform()

        # Test chi field by looking at moments.
        dx = 100.0 / c.shape[1]
        dy = 50.0 / c.shape[0]
        M = self._compute_moments(c, dx, dy, x=40.0, y=20.0)
        self.assertClose(M['0'], np.pi * 5.0 ** 2, rtol=1e-3)
        self.assertClose(M['1x'], 0.0, atol=1e-2)
        self.assertClose(M['1y'], 0.0, atol=1e-2)
        self.assertClose(M['2xx'], np.pi / 4 * 5.0 ** 4, rtol=1e-2)
        self.assertClose(M['2yy'], np.pi / 4 * 5.0 ** 4, rtol=1e-2)
        self.assertClose(M['2xy'], 0.0, atol=0.2)
