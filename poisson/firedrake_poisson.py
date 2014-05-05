from poisson import Poisson
from firedrake import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


class FiredrakePoisson(Poisson):
    series = {'np': op2.MPI.comm.size}

    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'mesh': {'marker': '+', 'linestyle': '-'},
                 'setup': {'marker': 'x', 'linestyle': '-'},
                 'matrix assembly': {'marker': '>', 'linestyle': '-'},
                 'rhs assembly': {'marker': '<', 'linestyle': '-'},
                 'solve': {'marker': 'D', 'linestyle': '-'}}

    def poisson(self, size=32, degree=1, dim=2, preassemble=True, pc='jacobi'):
        params = {'ksp_type': 'cg',
                  'pc_type': pc,
                  'ksp_rtol': 1e-6,
                  'ksp_atol': 1e-15}
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)

            # Define boundary condition
            bc = DirichletBC(V, 0.0, [3, 4])

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Function(V).interpolate(Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)"))
            g = Function(V).interpolate(Expression("sin(5*x[0])"))
            f.dat._force_evaluation()
            g.dat._force_evaluation()
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx + g*v*ds

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

if __name__ == '__main__':
    op2.init(log_level='WARNING')

    # Benchmark
    FiredrakePoisson().main(benchmark=True, save=None)

    # Profile
    regions = ['matrix assembly', 'rhs assembly', 'solve']
    FiredrakePoisson().profile(regions=regions)
