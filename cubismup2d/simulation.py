from libcubismup2d import SimulationData
import libcubismup2d as libcup2d

from typing import Any, List, Optional, Tuple, Union
import os

__all__ = ['Operator', 'Simulation', 'SimulationData']


def sanitize_arg(x: Any):
    if x is None:
        raise TypeError(x)
    elif isinstance(x, bool):
        x = int(x)
    return str(x)


class _FieldsProxy:
    __slots__ = ('data',)
    def __init__(self, data: SimulationData):
        self.data = data

    @property
    def chi(self):
        return self.data.chi

    @property
    def vel(self):
        return self.data.vel

    @property
    def vOld(self):
        return self.data.vOld

    @property
    def pres(self):
        return self.data.pres

    @property
    def tmpV(self):
        return self.data.tmpV

    @property
    def tmp(self):
        return self.data.tmp

    @property
    def uDef(self):
        return self.data.uDef

    @property
    def pOld(self):
        return self.data.pOld

    @property
    def Cs(self):
        return self.data.Cs


class Simulation(libcup2d._Simulation):
    def __init__(
            self,
            cells: Tuple[int, int],
            *,
            nlevels: int = 6,
            start_level: Optional[int] = None,
            rtol: float = 0.5,
            ctol: float = 0.1,
            extent: float = 1.0,
            cfl: float = 0.4,
            dt: float = 0.0,
            nu: float = 0.001,
            brinkman_lambda: float = 1e6,
            fdump: int = 0,
            tdump: float = 0.0,
            output_dir: str = 'output/',
            serialization_dir: Optional[str] = None,
            verbose: bool = True,
            mute_all: bool = False,
            ic: str = '',
            BCx: str = 'freespace',
            BCy: str = 'freespace',
            smagorinskyCoeff: float = 0.0,
            dumpCs: bool = False,
            bForcing: bool = False,
            forcingC: float = 4.0,
            forcingW: float = 4.0,
            cuda: bool = False,
            comm: Optional['mpi4py.MPI.Intracomm'] = None,
            argv: List[str] = []):
        """
        Arguments:
            ...
            nlevels: number of levels, use 1 for a uniform grid
            start_level: level at which the grid is initialized,
                         defaults to min(nlevels - 1, 3)
            ...
            cfl: (float) target CFL number for automatic dt
            dt: (float) manual time step (only if `cfl == 0.0`)
            ...
            serialization_dir: folder containing HDF5 files,
                               defaults to `os.path.join(output_dir, 'h5')`
            cuda: (bool) if True, use cuda_iterative Poisson solver
            argv: (list of strings) extra argv passed to CubismUP2D
        """
        if cfl != 0.0 and dt != 0.0:
            raise ValueError("Cannot specify both `cfl` and `dt`. To use "
                             "a fixed time step, set `cfl` to 0.")
        if not isinstance(nlevels, int) or nlevels < 1:
            raise ValueError(f"expected integer larger than 1, got {nlevels!r}")
        if len(cells) != 2:
            raise ValueError(f"expected 2 values, got {cells!r}")
        self.cells = cells
        if any(c % libcup2d.BLOCK_SIZE != 0 for c in cells):
            raise ValueError(f"number of cells must be a multiple of the block "
                             f"size of {libcup2d.BLOCK_SIZE}, got {cells!r}")
        if start_level is None:
            start_level = min(nlevels - 1, 3)
        if serialization_dir is None:
            serialization_dir = os.path.join(output_dir, 'h5')
        argv = [
            '-bpdx', cells[0] // libcup2d.BLOCK_SIZE,
            '-bpdy', cells[1] // libcup2d.BLOCK_SIZE,
            '-levelMax', nlevels,
            '-levelStart', start_level,
            '-Rtol', rtol,
            '-Ctol', ctol,
            '-extent', extent,
            '-CFL', cfl,
            '-dt', dt,
            '-nu', nu,
            '-lambda', brinkman_lambda,
            '-fdump', fdump,
            '-tdump', tdump,
            '-tend', 0.0,  # Specified through `simulate`.
            '-nsteps', 0,
            '-file', output_dir,
            '-serialization', serialization_dir,
            '-verbose', verbose,
            '-muteAll', mute_all,
            '-ic', ic,
            '-BC_x', BCx,
            '-BC_y', BCy,
            '-smagorinskyCoeff', smagorinskyCoeff,
            '-dumpCs', dumpCs,
            '-bForcing', bForcing,
            '-forcingCoefficient', forcingC,
            '-forcingWavenumber', forcingW,
            *(['-poissonSolver', 'cuda_iterative'] if cuda else []),
            *argv,
        ]
        argv = [sanitize_arg(arg) for arg in argv]

        if comm is not None:
            from mpi4py import MPI
            comm = MPI._addressof(comm)
        else:
            comm = 0

        # Ideally this should be in Cubism's dump function.
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(serialization_dir, exist_ok=True)
        libcup2d._Simulation.__init__(self, ['DUMMY'] + argv, comm)
        self._ops = []
        self.fields = _FieldsProxy(self.data)

    def insert_operator(self, op, *args, **kwargs):
        # We have to store an in-Python reference permanently.
        # https://github.com/pybind/pybind11/issues/1546
        # https://github.com/pybind/pybind11/pull/2839
        self._ops.append(op)
        super().insert_operator(op, *args, **kwargs)

    def simulate(self,
                 *,
                 nsteps: Optional[int] = None,
                 tend: Optional[float] = None):
        data: libcup2d.SimulationData = self.data
        data._nsteps = data.step + nsteps if nsteps is not None else 0
        data._tend = data.time + tend if tend is not None else 0.0
        super().simulate()


class Operator(libcup2d._Operator):
    __slots__ = ('sim',)
    def __init__(self, sim: Simulation, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        libcup2d._Operator.__init__(self, sim.data, name)
        self.sim = sim
