from matrix_assembly import MatrixAssembly
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinMatrixAssembly(MatrixAssembly):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'mesh': {'marker': '+', 'linestyle': '--'},
                 'setup': {'marker': 'x', 'linestyle': '--'},
                 'assembly': {'marker': '>', 'linestyle': '--'},
                 'reassembly': {'marker': '<', 'linestyle': '--'},
                 'assembly bcs': {'marker': '^', 'linestyle': '--'},
                 'reassembly bcs': {'marker': 'v', 'linestyle': '--'}}

    def matrix_assembly(self, size=32, degree=1, dim=2, fs='scalar'):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()
        with self.timed_region('setup'):
            FS = {'scalar': FunctionSpace, 'vector': VectorFunctionSpace}[fs]
            V = FS(mesh, "Lagrange", degree)

            # Define Dirichlet boundary (x = 0 or x = 1)
            def boundary(x):
                return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

            # Define boundary condition
            u0 = {'scalar': Constant(0.0), 'vector': Constant((0.0,)*dim)}[fs]
            bc = DirichletBC(V, u0, boundary)

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            a = inner(grad(u), grad(v))*dx

            # Compute solution
            u = Function(V)
        with self.timed_region('assembly'):
            A = assemble(a)
        with self.timed_region('reassembly'):
            A = assemble(a, tensor=A)
        with self.timed_region('assembly bcs'):
            A = assemble(a)
            bc.apply(A)
        with self.timed_region('reassembly bcs'):
            A = assemble(a, tensor=A)
            bc.apply(A)

if __name__ == '__main__':
    set_log_active(False)

    # Benchmark
    DolfinMatrixAssembly().main(benchmark=True, save=None)

    # Profile
    regions = ['assembly', 'reassembly', 'assembly bcs', 'reassembly bcs']
    DolfinMatrixAssembly().profile(regions=regions)
