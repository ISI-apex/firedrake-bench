from poisson import Poisson
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}
initial = {2: "32*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])",
           3: "48*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}
analytical = {2: "cos(4*pi*x[0])*sin(4*pi*x[1])",
              3: "cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinPoisson(Poisson):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}

    def poisson(self, size=32, degree=1, dim=2, preassemble=True, pc='amg', print_norm=True):
        params = {'linear_solver': 'cg',
                  'preconditioner': pc}
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)
            print '[%d]' % MPI.rank(mpi_comm_world()), 'DOFs:', V.dofmap().global_dimension()

            # Define Dirichlet boundary (x = 0 or x = 1)
            def boundary(x):
                return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

            # Define boundary condition
            u0 = Constant(0.0)
            bc = DirichletBC(V, u0, boundary)

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Expression(initial[dim])
            a = inner(grad(u), grad(v))*dx
            L = f*v*dx

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
        a = Expression(analytical[dim])
        l2 = sqrt(assemble(dot(u - a, u - a) * dx))
        if print_norm and MPI.rank(mpi_comm_world()) == 0:
            print 'L2 error norm:', l2
        t = timings(True)
        for task in ['Assemble cells', 'Build sparsity', 'DirichletBC apply',
                     'PETSc Krylov solver']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinPoisson().main()
