from poisson import Poisson
from pybench import timed
from dolfin import *

initial = {2: "32*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])",
           3: "48*pi*pi*cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}
analytical = {2: "cos(4*pi*x[0])*sin(4*pi*x[1])",
              3: "cos(4*pi*x[0])*sin(4*pi*x[1])*cos(4*pi*x[2])"}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


@timed
def make_mesh(dim, x):
    mesh = UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)
    mesh.init()
    return mesh


class DolfinPoisson(Poisson):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}
    meta = {'dolfin_version': dolfin_version(),
            'dolfin_commit': git_commit_hash()}
    meshes = {}

    def poisson(self, size=32, degree=1, dim=3, preassemble=True, weak=False,
                print_norm=True, verbose=False, pc='amg',
                strong_threshold=0.75, agg_nl=2, max_levels=25):
        if weak:
            size = int((1e4*MPI.size(mpi_comm_world()))**(1./dim))
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.series['degree'] = degree
        self.meta['cells'] = 6*size**dim
        self.meta['vertices'] = (size+1)**dim
        params = {'linear_solver': 'cg',
                  'preconditioner': pc}
        # Tune AMG parameters
        PETScOptions.set('pc_hypre_boomeramg_strong_threshold', strong_threshold)
        PETScOptions.set('pc_hypre_boomeramg_agg_nl', agg_nl)
        PETScOptions.set('pc_hypre_boomeramg_max_levels', max_levels)
        if verbose:
            PETScOptions.set('pc_hypre_boomeramg_print_statistics', True)
            PETScOptions.set('ksp_view', True)
            PETScOptions.set('ksp_monitor', True)

        if (dim, size) in self.meshes:
            t_, mesh = self.meshes[dim, size]
        else:
            t_, mesh = make_mesh(dim, size)
            self.meshes[dim, size] = t_, mesh
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)
            if verbose:
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
                self.meta['dofs'] = A.size(0)
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
