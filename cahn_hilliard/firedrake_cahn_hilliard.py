# Can't use relative imports outside of a package,
# but don't want to have to set PYTHONPATH all the time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Needed only for UCX (UCX does not support threaded mode in OpenMPI)
#from mpi4py import rc
#rc.thread_level = "single"

from cahn_hilliard import CahnHilliard, lmbda, dt, theta
from firedrake_cahn_hilliard_problem import CahnHilliardProblem
from firedrake import *
#from pyop2.profiling import get_timers, timing

#import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("RANK: ", comm.rank)

from firedrake_common import FiredrakeBenchmark

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True


class FiredrakeCahnHilliard(FiredrakeBenchmark, CahnHilliard):

    warmups = 0
    repeats = 1

    def cahn_hilliard(self, size=96, steps=1, degree=1, pc='fieldsplit',
                      inner_ksp='preonly', ksp='gmres', maxit=1, weak=False,
                      measure_overhead=False, save=False, compute_norms=True,
                      verbose=False):

        if weak:
            self.series['weak'] = size
            size = int((size*comm.size)**0.5)
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.meta['cells'] = 2*size**2
        self.meta['vertices'] = (size+1)**2

        t_, mesh = self.make_mesh(size)
        self.register_timing('mesh', t_)

        with self.timed_region('setup'):
            u, u0, solver = CahnHilliardProblem.do_setup(mesh, pc=pc,
                    degree=degree, inner_ksp=inner_ksp, maxit=maxit,
                    theta=theta, dt=dt, lmbda=lmbda,
                    ksp=ksp, inner_ksp=inner_ksp, verbose=verbose)

        # Output file
        if save:
            file = File("vtk/firedrake_cahn_hilliard_%d.pvd" % size)
        else:
            file = None

        if measure_overhead:
            CahnHilliardProblem.do_measure_overhead(u0, solver)
            #print("Assembly overhead:", timing("Assemble cells", total=False))
            #prin("Solver overhead:", timing("SNES solver execution", total=False))
            print("Assembly overhead: TODO")
            print("Solver overhead: TODO")
            return

        with self.timed_region('timestepping'):
            CahnHilliardProblem.do_solve(u, u0, solver, steps,
                    out_file=file)

        #for task, timer in get_timers(reset=True).items():
        #    self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    # Benchmark
    #FiredrakeCahnHilliard().main(run=True)
    b = FiredrakeCahnHilliard()
    b.main(benchmark=True)
    print(b.result)
    for func, t in b.result['timings'].items():
        print("%24s: %.6f" % (func, t))
    #FiredrakeCahnHilliard().main(profile=True)

    # Output VTU files
    # FiredrakeCahnHilliard().cahn_hilliard(save=True)
