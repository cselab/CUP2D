import libcubismup2d as lib
from .simulation import Simulation, sanitize_arg

from typing import Any, Dict, Optional, Tuple

__all__ = ['Disk', 'HalfDisk', 'Ellipse', 'Rectangle', 'StefanFish']

def _init(
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
        vel: Tuple[float, float] = (0.0, 0.0),
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
        'xvel': vel[0],
        'yvel': vel[1],
        'angvel': omega,
        'dumpSurf': dump_surface,
        'timeForced': time_forced,
    }
    conflict = kwargv.keys() & _kwargv.keys()
    assert not conflict, conflict

    # Arrange in the format that the C++-side parser expects.
    kwargv = [f'{k}={sanitize_arg(v)}' for k, v in kwargv.items()]
    _kwargv = [f'{k}={sanitize_arg(v)}'
               for k, v in _kwargv.items() if v is not None]
    argv = ' '.join(kwargv) + ' ' + ' '.join(_kwargv)
    cls.__init__(shape, sim.data, argv, center)


class Disk(lib._Disk):
    __slots__ = ()
    def __init__(self, sim: Simulation, r: float, *, tAccel: float = 0.0, **kwargs):
        _init(lib._Disk, self, sim, dict(radius=r, tAccel=tAccel), **kwargs)


class HalfDisk(lib._HalfDisk):
    __slots__ = ()
    def __init__(self, sim: Simulation, r: float, *, tAccel: float = 0.0, **kwargs):
        _init(lib._HalfDisk, self, sim, dict(radius=r, tAccel=tAccel), **kwargs)


class Ellipse(lib._Ellipse):
    __slots__ = ()
    def __init__(self, sim: Simulation, a: float, b: float, **kwargs):
        """Construct an Ellipse with semiaxes `a` and `b`."""
        _init(lib._Ellipse, self, sim, dict(semiAxisX=a, semiAxisY=b), **kwargs)


class Rectangle(lib._Rectangle):
    __slots__ = ()
    def __init__(self, sim: Simulation, a: float, b: float, **kwargs):
        """Construct a Rectangle with sides `a` and `b`."""
        _init(lib._Rectangle, self, sim, dict(extentX=a, extentY=b), **kwargs)


class StefanFish(lib._StefanFish):
    __slots__ = ()
    def __init__(self, sim: Simulation, pid: int, pidpos: int, **kwargs):
        """Construct a StefanFish with required PID control `pid` and `pidpos`."""
        _init(lib._StefanFish, self, sim, dict(bCorrectTrajectory=pid, bCorrectPosition=pidpos), **kwargs)
