import libcubismup2d as libcup2d

from typing import Optional, Tuple, Union
import os

__all__ = ['Operator', 'Simulation']

class _FieldProperty:
    """Lazy attribute accessor."""
    __slots__ = ('_name',)

    def __set_name__(self, owner, name: str):
        self._name = name

    def __get__(self, obj, objtype=None) -> \
            Union[libcup2d.ScalarGrid, libcup2d.VectorGrid]:
        return getattr(obj.sim, self._name)


class _FieldsProxy:
    __slots__ = ('sim',)
    def __init__(self, sim: libcup2d._SimulationData):
        self.sim = sim

    chi = _FieldProperty()
    vel = _FieldProperty()
    vOld = _FieldProperty()
    pres = _FieldProperty()
    tmpV = _FieldProperty()
    tmp = _FieldProperty()
    uDef = _FieldProperty()
    pold = _FieldProperty()


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
            cfl: float = 0.1,
            nu: float = 0.001,
            brinkman_lambda: float = 1e6,
            fdump: int = 0,
            tdump: float = 0.0,
            output_dir: str = 'output/',
            serialization_dir: Optional[str] = None,
            verbose: bool = True,
            comm: Optional['mpi4py.MPI.Intracomm'] = None):
        """
        Arguments:
            ...
            nlevels: number of levels, set to 1 for a uniform grid
            start_level: level at which the grid is initialized,
                         defaults to min(nlevels - 1, 3)
            ...
            serialization_dir: folder containing HDF5 files,
                               defaults to `os.path.join(output_dir, 'h5')`
        """
        assert nlevels >= 1, nlevels
        if start_level is None:
            start_level = min(nlevels - 1, 3)
        if serialization_dir is None:
            serialization_dir = os.path.join(output_dir, 'h5')
        assert cells[0] % libcup2d.BLOCK_SIZE == 0, cells[0]
        assert cells[1] % libcup2d.BLOCK_SIZE == 0, cells[1]
        argv = [
            '-bpdx', cells[0] // libcup2d.BLOCK_SIZE,
            '-bpdy', cells[1] // libcup2d.BLOCK_SIZE,
            '-levelMax', nlevels,
            '-levelStart', start_level,
            '-Rtol', rtol,
            '-Ctol', ctol,
            '-extent', extent,
            '-CFL', cfl,
            '-nu', nu,
            '-lambda', brinkman_lambda,
            '-fdump', fdump,
            '-tdump', tdump,
            '-tend', 0.0,  # Specified through `simulate`.
            '-nsteps', 0,
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
        libcup2d._Simulation.__init__(self, ['DUMMY'] + argv, comm)
        self._ops = []
        self.fields = _FieldsProxy(self.sim)

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
        sim: libcup2d.SimulationData = self.sim
        sim._nsteps = sim.step + nsteps if nsteps is not None else 0
        sim._tend = sim.time + tend if tend is not None else 0.0
        super().simulate()


class Operator(libcup2d._Operator):
    def __init__(self, sim: Simulation, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        libcup2d._Operator.__init__(self, sim.sim, name)
