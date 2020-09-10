from copy import copy

from firedrake import *
#from firedrake import __version__ as firedrake_version
#from firedrake.utils import memoize
#from pyop2 import __version__ as pyop2_version
#from pyop2.profiling import tic, toc

from pybench import timed

import sys
#from common import get_petsc_version


class FiredrakeBenchmark(object):
    #series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}
    series = {'np': 1, 'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"]}
            #'firedrake': firedrake_version,
            #'pyop2': pyop2_version}
            #'petsc_version': get_petsc_version()}

    #@memoize
    @timed
    def make_mesh(self, x, dim=2):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)

    def lhs_overhead(self, a, bcs=None, N=1000):
        A = assemble(a, bcs=bcs)
        A.M
        #tic('matrix assembly')
        for _ in range(N):
            assemble(a, tensor=A, bcs=bcs).M
        #return toc('matrix assembly')/N
        return 0

    def lhs_ffc_overhead(self, a, bcs=None, N=1000):
        #tic('matrix assembly')
        for _ in range(N):
            # Need to create new copies of the forms, since kernels are cached
            assemble(copy(a), bcs=bcs).M
        #return toc('matrix assembly')/N
        return 0

    def rhs_overhead(self, L, bcs=None, N=1000):
        b = assemble(L, bcs=bcs)
        #tic('rhs assembly')
        for _ in range(N):
            assemble(L, tensor=b, bcs=bcs).dat.data_ro
        #return toc('rhs assembly')/N
        return 0

    def rhs_ffc_overhead(self, L, bcs=None, N=1000):
        #tic('rhs assembly')
        for _ in range(N):
            # Need to create new copies of the forms, since kernels are cached
            assemble(copy(L), bcs=bcs).dat.data_ro
        #return toc('rhs assembly')/N
        return 0
