
# for some reason I have to do this... IDK why or how I figured this out lol
import jax
if jax.__version__ == '0.7.0':
    import jax.experimental.pjit
    from jax.extend.core.primitives import jit_p
    jax.experimental.pjit.pjit_p = jit_p
# jax._src.xla_bridge._check_cuda_versions()
from . import models
from . import utils
from . import result
from . import experimental
