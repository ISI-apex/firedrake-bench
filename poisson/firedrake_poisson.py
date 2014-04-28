from poisson import Poisson
from firedrake import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


class FiredrakePoisson(Poisson):
    series = {'np': op2.MPI.comm.size}

    plotstyle = {'total': {'color': 'black',
                           'marker': '*',
                           'linestyle': '-'},
                 'mesh': {'color': 'blue',
                          'marker': '+',
                          'linestyle': '-'},
                 'setup': {'color': 'green',
                           'marker': 'x',
                           'linestyle': '-'},
                 'matrix assembly': {'color': 'cyan',
                                     'marker': '>',
                                     'linestyle': '-'},
                 'rhs assembly': {'color': 'magenta',
                                  'marker': '<',
                                  'linestyle': '-'},
                 'solve': {'color': 'red',
                           'marker': 'D',
                           'linestyle': '-'}}

    def poisson(self, size=32, degree=1, dim=2):
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
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx + g*v*ds

            # Compute solution
            u = Function(V)
        with self.timed_region('matrix assembly'):
            A = assemble(a, bcs=bc)
        with self.timed_region('rhs assembly'):
            b = assemble(L)
            bc.apply(b)
        with self.timed_region('solve'):
            solve(A, u, b, solver_parameters={'ksp_type': 'cg',
                                              'pc_type': 'jacobi',
                                              'ksp_rtol': 1e-6,
                                              'ksp_atol': 1e-15})
            u.dat.data

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    FiredrakePoisson().main(benchmark=True, save=None)
