from poisson import Poisson
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinPoisson(Poisson):
    series = {'np': MPI.size(mpi_comm_world())}

    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'mesh': {'marker': '+', 'linestyle': '--'},
                 'setup': {'marker': 'x', 'linestyle': '--'},
                 'matrix assembly': {'marker': '>', 'linestyle': '--'},
                 'rhs assembly': {'marker': '<', 'linestyle': '--'},
                 'solve': {'marker': 'D', 'linestyle': '--'}}

    def poisson(self, size=32, degree=1, dim=2, preassemble=True, pc='amg'):
        params = {'linear_solver': 'cg',
                  'preconditioner': pc}
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)

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
        if preassemble:
            with self.timed_region('matrix assembly'):
                A = assemble(a)
                bc.apply(A)
            with self.timed_region('rhs assembly'):
                b = assemble(L)
                bc.apply(b)
            with self.timed_region('solve'):
                solve(A, u.vector(), b, 'cg', pc)
        else:
            with self.timed_region('solve'):
                solve(a == L, u, bcs=bc, solver_parameters=params)
        t = timings(True)
        for task in ['Assemble cells', 'Assemble exterior facets',
                     'Build sparsity', 'DirichletBC apply', 'PETSc Krylov solver']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)

    DolfinPoisson().main()
