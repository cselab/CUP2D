from base import TestCase, TestSimulation, cup2d

class TestSimulationCase(TestCase):
    def test_nsteps(self):
        cnt = [0]
        class TestOperator(cup2d.Operator):
            def __call__(self, dt: float):
                cnt[0] += 1

        sim = TestSimulation(cells=(64, 64), nlevels=1)
        sim.insert_operator(TestOperator(sim))
        sim.init()
        sim.simulate(nsteps=3)
        self.assertEqual(cnt[0], 3)

    def test_tend(self):
        dts = []
        class TestOperator(cup2d.Operator):
            def __call__(self, dt: float):
                dts.append(dt)

        sim = TestSimulation(cells=(64, 64), nlevels=1, extent=100.0)
        sim.add_shape(cup2d.Disk(sim, r=15.0, center=(40.0, 30.0)))
        sim.insert_operator(TestOperator(sim))
        sim.init()
        sim.simulate(tend=1.2)
        self.assertGreaterEqual(sum(dts), 1.2 - 1e-8)

    def test_manually_adapt(self):
        # Test that nothing crashes here.
        sim = TestSimulation(cells=(64, 64), start_level=1, nlevels=3)
        sim.init()
        sim.simulate(nsteps=10)
        sim.adapt_mesh()
        sim.simulate(nsteps=10)
