#!/usr/bin/env python3

"""Flow around a stefanfish. Output files are stored in output/."""

import cubismup2d as cup2d
import numpy as np


sim = cup2d.Simulation(cells=(1024, 512), nlevels=1, start_level=0, extent=2.0, tdump=0.1, argv=['-shapes', 'stefanfish L=0.2 T=1 xpos=0.6 bFixed=1'])
sim.init()
sim.simulate(tend=10.0)
