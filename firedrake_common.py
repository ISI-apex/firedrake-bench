from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
from pyop2 import __version__ as pyop2_version

from pybench import timed

from common import get_petsc_version


class FiredrakeBenchmark(object):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"],
            'firedrake': firedrake_version,
            'pyop2': pyop2_version,
            'petsc_version': get_petsc_version()}

    @memoize
    @timed
    def make_mesh(self, dim, x):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)
