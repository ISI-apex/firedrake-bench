from copy import copy

from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
from pyop2 import __version__ as pyop2_version
from pyop2.profiling import tic, toc

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

    def lhs_overhead(self, a, bcs=None):
        tic('matrix assembly')
        for _ in range(1000):
            # Need to create new copies of the forms, since kernels are cached
            assemble(copy(a), bcs=bcs).M
        return toc('matrix assembly')/1000

    def rhs_overhead(self, L, bcs=None):
        tic('rhs assembly')
        for _ in range(1000):
            # Need to create new copies of the forms, since kernels are cached
            assemble(copy(L), bcs=bcs).dat._force_evaluation()
        return toc('rhs assembly')/1000
