from poisson import Poisson
from firedrake import *
# from pyop2.coffee.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers

from firedrake_common import FiredrakeBenchmark

initial = {2: "32*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])",
           3: "48*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}
analytical = {2: "cos(4*pi*x[0])*sin(4*pi*x[1])",
              3: "cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}

parameters["assembly_cache"]["enabled"] = False
parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
# Vectorization appears to degrade performance for p2
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


class FiredrakePoisson(FiredrakeBenchmark, Poisson):

    def poisson(self, size=32, degree=1, dim=3, preassemble=True, weak=False,
                print_norm=True, verbose=False, measure_overhead=False,
                pc='hypre', strong_threshold=0.75, agg_nl=2, max_levels=25):
        if weak:
            self.series['weak'] = size
            size = int((size*op2.MPI.comm.size)**(1./dim))
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.series['degree'] = degree
        self.meta['cells'] = 6*size**dim
        self.meta['vertices'] = (size+1)**dim
        params = {'ksp_type': 'cg',
                  'pc_type': pc,
                  'pc_hypre_type': 'boomeramg',
                  'pc_hypre_boomeramg_strong_threshold': strong_threshold,
                  'pc_hypre_boomeramg_agg_nl': agg_nl,
                  'pc_hypre_boomeramg_max_levels': max_levels,
                  'ksp_rtol': 1e-6,
                  'ksp_atol': 1e-15}
        if verbose:
            params['pc_hypre_boomeramg_print_statistics'] = True
            params['ksp_view'] = True
            params['ksp_monitor'] = True
        t_, mesh = self.make_mesh(size, dim)
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)
            if verbose:
                print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size

            # Define boundary condition
            bc = DirichletBC(V, 0.0, [3, 4])

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Function(V).interpolate(Expression(initial[dim]))
            f.dat.data_ro
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx

            # Compute solution
            u = Function(V)
        if measure_overhead:
            print "Matrix assembly overhead:", self.lhs_ffc_overhead(a, bc)
            print "RHS assembly overhead:", self.rhs_ffc_overhead(L, bc)
            return
        if preassemble:
            with self.timed_region('matrix assembly'):
                A = assemble(a, bcs=bc)
                A.M
            self.meta['dofs'] = A.M.handle.sizes[0][1]
            with self.timed_region('rhs assembly'):
                b = assemble(L)
                bc.apply(b)
                b.dat.data_ro
            with self.timed_region('solve'):
                solve(A, u, b, solver_parameters=params)
                u.dat.data_ro
        else:
            with self.timed_region('solve'):
                solve(a == L, u, bcs=[bc], solver_parameters=params)
                u.dat.data_ro

        # Analytical solution
        a = Function(V).interpolate(Expression(analytical[dim]))
        l2 = sqrt(assemble(dot(u - a, u - a) * dx))
        if print_norm and op2.MPI.comm.rank == 0:
            print 'L2 error norm:', l2
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    FiredrakePoisson().main()
