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

    def test_nsteps(self):
        cnt = [0]
        class TestOperator(cup2d.Operator):
            def __call__(self, dt: float):
                cnt[0] += 1

        sim = cup2d.Simulation(cells=(64, 64), nlevels=1)
        sim.insert_operator(TestOperator(sim))
        sim.init()
        sim.simulate(nsteps=3)
        self.assertEqual(cnt[0], 3)

    def test_tend(self):
        dts = []
        class TestOperator(cup2d.Operator):
            def __call__(self, dt: float):
                dts.append(dt)

        sim = cup2d.Simulation(cells=(64, 64), nlevels=1, extent=100.0)
        sim.add_shape(cup2d.Disk(sim, r=15.0, center=(40.0, 30.0)))
        sim.insert_operator(TestOperator(sim))
        sim.init()
        sim.simulate(tend=1.2)
        self.assertClose(sum(dts), 1.2, rtol=1e-10)

        # Try again with only the first dt + eps. Assert that the second dt=eps
        # time step does not run unnecessarily.
        first_dt = dts[0]
        dts.clear()
        sim = cup2d.Simulation(cells=(64, 64), nlevels=1, extent=100.0)
        sim.add_shape(cup2d.Disk(sim, r=15.0, center=(40.0, 30.0)))
        sim.insert_operator(TestOperator(sim))
        sim.init()
        sim.simulate(tend=first_dt + 1e-16)
        self.assertEqual(len(dts), 1)
