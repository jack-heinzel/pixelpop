from .nearest_neighbor import *
from .data import *

# Posterior collection depends on the optional `gwpopulation_pipe` package. Expose it
# when available, but don't make core `import pixelpop` require it.
try:
    from . import collection
except ImportError:
    collection = None
