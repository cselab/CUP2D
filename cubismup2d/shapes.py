from cubismup2d.simulation import Simulation
from libcubismup2d import _Shape, _Disk

from typing import Any, Dict, Optional, Tuple

__all__ = ['Disk']

def _init_shape(
        cls: type,
        shape,
        sim: Simulation,
        kwargv: Dict[str, Any],
        *,
        center: Tuple[float, float],
        angle: float = 0.0,
        rhoS: float = 1.0,  # What is this?
        fixed: bool = False,
        fixedx: Optional[bool] = None,
        fixedy: Optional[bool] = None,
        forced: bool = False,
        forcedx: Optional[bool] = None,
        forcedy: Optional[bool] = None,
        block_angle: Optional[bool] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
        omega: float = 0.0,
        dump_surface: int = 0,
        time_forced: float = 1e100):
    """Helper function for initializing shapes.

    Used instead of multiple inheritance to avoid the diamond inheritance
    pattern which may cause trouble.
    """
    _kwargv = {
        'angle': angle,
        'rhoS': rhoS,
        'bFixed': fixed,
        'bFixedx': fixedx,
        'bFixedy': fixedy,
        'bForced': forced,
        'bForcedx': forcedx,
        'bForcedy': forcedy,
        'xvel': velocity[0],
        'yvel': velocity[1],
        'angvel': omega,
        'dumpSurf': dump_surface,
        'timeForced': time_forced,
    }
    conflict = kwargv.keys() & _kwargv.keys()
    assert not conflict, conflict

    # Arrange in the format that the C++-side parser expects.
    kwargv = [f'{k}={v}' for k, v in kwargv.items()]
    _kwargv = [f'{k}={v}' for k, v in _kwargv.items() if v is not None]
    argv = ' '.join(kwargv) + ' ' + ' '.join(_kwargv)
    cls.__init__(shape, sim.sim, argv, center)


class Disk(_Disk):
    def __init__(
            self,
            sim: Simulation,
            *,
            r: float,
            tAccel: float = 0.0,
            **kwargs):
        _init_shape(_Disk, self, sim, dict(radius=r, tAccel=tAccel), **kwargs)
