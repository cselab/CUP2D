#!/usr/bin/env python3

"""Flow behind a rectangle. Output files are stored in output/."""

import cubismup2d as cup2d
import numpy as np


sim = cup2d.Simulation(cells=(128, 64), nlevels=1, start_level=0,
                       extent=100.0, tdump=0.1)
rectangle = cup2d.Rectangle(
        sim, a=10.0, b=15.0, center=(50.0, 25.0), vel=(10.0, 0.0),
        fixed=True, forced=True)
sim.add_shape(rectangle)
sim.init()
sim.simulate(tend=10.0)
