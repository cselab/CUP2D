import libcubismup2d as libcup2d

from typing import Optional, Tuple
import os

__all__ = ['Simulation']

class Simulation(libcup2d.Simulation):
    def __init__(
            self,
            cells: Tuple[int, int],
            *,
            max_level: int = 6,
            start_level: int = 3,
            rtol: float = 0.5,
            ctol: float = 0.1,
            extent: float = 1.0,
            cfl: float = 0.1,
            nu: float = 0.001,
            brinkman_lambda: float = 1e6,
            fdump: int = 0,
            tdump: float = 0.0,
            tend: float = 0.0,
            max_steps: int = 0,
            output_dir: str = 'output/',
            serialization_dir: str = 'output/h5/',
            verbose: bool = True,
            comm: Optional['mpi4py.MPI.Intracomm'] = None):
        assert cells[0] % libcup2d.BLOCK_SIZE == 0, cells[0]
        assert cells[1] % libcup2d.BLOCK_SIZE == 0, cells[1]
        argv = [
            '-bpdx', cells[0] // libcup2d.BLOCK_SIZE,
            '-bpdy', cells[1] // libcup2d.BLOCK_SIZE,
            '-levelMax', max_level,
            '-levelStart', start_level,
            '-Rtol', rtol,
            '-Ctol', ctol,
            '-extent', extent,
            '-CFL', cfl,
            '-nu', nu,
            '-lambda', brinkman_lambda,
            '-fdump', fdump,
            '-tdump', tdump,
            '-tend', tend,
            '-nsteps', max_steps,
            '-file', output_dir,
            '-serialization', serialization_dir,
        ]
        argv = [str(arg) for arg in argv]

        if comm is not None:
            from mpi4py import MPI
            comm = MPI._addressof(comm)
        else:
            comm = 0

        # Ideally this should be in Cubism's dump function.
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(serialization_dir, exist_ok=True)
        libcup2d.Simulation.__init__(self, ['DUMMY'] + argv, comm)
