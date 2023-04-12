


from . import fun
from . import functional
from . import colloc
from . import linalg
from . import newton

from fun import Fun


# What is this?
def test(level=1, verbosity=1):
    from numpy.testing import Tester
    return Tester().test(level, verbosity)


