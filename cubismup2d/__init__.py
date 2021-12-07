try:
    from libcubismup2d import *
except ModuleNotFoundError:
    # TODO: configure install in cmake
    raise ImportError("libcubismup2d not found, add the build folder to PYTHONPATH")

from .shapes import *
from .simulation import *
