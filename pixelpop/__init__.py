
# for some reason I have to do this... IDK why or how I figured this out lol
import jax

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'
    
from . import models
from . import utils
from . import result
from . import experimental
