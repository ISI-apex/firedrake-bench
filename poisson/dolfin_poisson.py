from poisson import Poisson
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


class DolfinPoisson(Poisson):

    plotstyle = {'total': {'color': 'black',
                           'marker': '*',
                           'linestyle': '--'},
                 'mesh': {'color': 'blue',
                          'marker': '+',
                          'linestyle': '--'},
                 'setup': {'color': 'green',
                           'marker': 'x',
                           'linestyle': '--'},
                 'matrix assembly': {'color': 'cyan',
                                     'marker': '>',
                                     'linestyle': '--'},
                 'rhs assembly': {'color': 'magenta',
                                  'marker': '<',
                                  'linestyle': '--'},
                 'solve': {'color': 'red',
                           'marker': 'D',
                           'linestyle': '--'}}

    def poisson(self, size, degree=1, dim=2):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", 1)

            # Define Dirichlet boundary (x = 0 or x = 1)
            def boundary(x):
                return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

            # Define boundary condition
            u0 = Constant(0.0)
            bc = DirichletBC(V, u0, boundary)

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
            g = Expression("sin(5*x[0])")
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx + g*v*ds

            # Compute solution
            u = Function(V)
        with self.timed_region('matrix assembly'):
            A = assemble(a)
            bc.apply(A)
        with self.timed_region('rhs assembly'):
            b = assemble(L)
            bc.apply(b)
        with self.timed_region('solve'):
            solve(A, u.vector(), b, "cg", "default")

if __name__ == '__main__':
    set_log_active(False)
    b = DolfinPoisson()
    b.run()
    b.plot(xaxis='size')
