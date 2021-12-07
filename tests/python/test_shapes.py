from base import TestCase
import cubismup2d as cup2d

import os

class TestShapes(TestCase):
    def test_disk(self):
        sim = cup2d.Simulation(cells=(64, 64), max_level=4, start_level=1,
                               extent=100.0, tend=10.0, fdump=1)
        disk = cup2d.Disk(sim, r=5.0, center=(30.0, 25.0))
        sim.add_shape(disk)
        sim.init()
        sim.simulate()
