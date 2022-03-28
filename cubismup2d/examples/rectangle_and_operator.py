#!/usr/bin/env python3

"""Flow behind a rectangle, with a custom operator.
Output files are stored in output/."""

import cubismup2d as cup2d
import numpy as np

class CustomOperator(cup2d.Operator):
    def __call__(self, dt: float):
        data: cup2d.SimulationData = self.sim.data

        # Accessing blocks of a field.
        for block in data.chi.blocks:
            b = block.data # modifying b will modify the data in CUP
            print(f"chi block: {block}    "
                  f"numpy array: (shape={b.shape}, dtype={b.dtype}, sum={b.sum()})")

        # Get the whole field as a large uniform matrix (this copies the whole
        # grid). Note that the order of axes is chi[y, x], not chi[x, y]!
        chi = data.chi.to_uniform()
        print(f"whole chi: shape={chi.shape} sum={chi.sum()})")

        # Updating the field (currently works only for uniform grids).
        # data.chi.load_uniform(some_matrix)

        # Print some metadata.
        print(f"-------------------> dt={dt} uinf=({data.uinfx} {data.uinfy})")


sim = cup2d.Simulation(cells=(128, 64), nlevels=1, start_level=0,
                       extent=100.0, tdump=0.1)
rectangle = cup2d.Rectangle(
        sim, a=10.0, b=15.0, center=(50.0, 25.0), vel=(10.0, 0.0),
        fixed=True, forced=True)
sim.add_shape(rectangle)
sim.init()
sim.insert_operator(CustomOperator(sim), after='advDiff')
sim.simulate(tend=10.0)
