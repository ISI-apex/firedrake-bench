from poisson import Poisson
from pybench import timed
from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
# from pyop2.coffee.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers
from pyop2 import __version__ as pyop2_version

initial = {2: "32*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])",
           3: "48*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}
analytical = {2: "cos(4*pi*x[0])*sin(4*pi*x[1])",
              3: "cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
# Vectorization appears to degrade performance for p2
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)

Poisson.meta.update({'coffee': parameters["coffee"],
                     'firedrake': firedrake_version,
                     'pyop2': pyop2_version})


class FiredrakePoisson(Poisson):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}

    @memoize
    @timed
    def make_mesh(self, dim, x):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)

    def poisson(self, size=32, degree=1, dim=2, preassemble=True, pc='hypre', print_norm=True):
        params = {'ksp_type': 'cg',
                  'pc_type': pc,
                  'pc_hypre_type': 'boomeramg',
                  'pc_hypre_boomeramg_strong_threshold': 0.75,
                  'pc_hypre_boomeramg_agg_nl': 2,
                  'ksp_rtol': 1e-6,
                  'ksp_atol': 1e-15}
        t_, mesh = self.make_mesh(dim, size)
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)
            print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size

            # Define boundary condition
            bc = DirichletBC(V, 0.0, [3, 4])

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Function(V).interpolate(Expression(initial[dim]))
            f.dat._force_evaluation()
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx

            # Compute solution
            u = Function(V)
        if preassemble:
            with self.timed_region('matrix assembly'):
                A = assemble(a, bcs=bc)
                A.M
            with self.timed_region('rhs assembly'):
                b = assemble(L)
                bc.apply(b)
                b.dat._force_evaluation()
            with self.timed_region('solve'):
                solve(A, u, b, solver_parameters=params)
                u.dat._force_evaluation()
        else:
            with self.timed_region('solve'):
                solve(a == L, u, bcs=[bc], solver_parameters=params)
                u.dat._force_evaluation()

        # Analytical solution
        a = Function(V).interpolate(Expression(analytical[dim]))
        l2 = sqrt(assemble(dot(u - a, u - a) * dx))
        if print_norm and op2.MPI.comm.rank == 0:
            print 'L2 error norm:', l2
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakePoisson().main()
