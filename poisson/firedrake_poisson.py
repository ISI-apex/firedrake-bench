from pybench import Benchmark
from firedrake import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


class FiredrakePoisson(Benchmark):

    params = {'degree': range(1, 4),
              'size': [2**x for x in range(4, 7)],
              'dim': [3]}
    method = 'poisson'

    def poisson(self, size, degree=1, dim=2):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", 1)

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
            solve(A, u, b, solver_parameters={"ksp_type": "cg"})
            u.dat.data

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    b = FiredrakePoisson()
    b.run()
    b.plot(xaxis='size')
