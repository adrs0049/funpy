
from . import support

from . import cheb

from . import trig

from . import fun

from . import functional

from . import vectorspaces
from . import colloc
from . import support
from . import linalg
from . import newton
from . import colloc

from .fun import Fun
from .fun import roots
from .fun import minandmax
from .fun import prolong
from .fun import plot
from .fun import plot_values
from .fun import plotcoeffs_trig
from .fun import plotcoeffs_cheb
from .fun import plotcoeffs
from .fun import zeros, ones, random, random_decay, asfun
from .fun import qr, norm, norm2, wkpnorm, h1norm, h2norm, normh
from .fun import sturm_norm, sturm_norm_alt
from .fun import normalize, innerw

from .functional import Functional

from .cbcode import npcode, spcode, cpcode
from .odecode import odecode

from .mapping import Mapping
from . ultra import ultra2ultra

# What is this?
def test(level=1, verbosity=1):
    from numpy.testing import Tester
    return Tester().test(level, verbosity)


