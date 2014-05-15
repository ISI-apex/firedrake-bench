from assembly import Assembly
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinAssembly(Assembly):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'mesh': {'marker': '+', 'linestyle': '--'},
                 'setup': {'marker': 'x', 'linestyle': '--'},
                 'mass premult 0': {'marker': '>', 'linestyle': '--'},
                 'mass premult 1': {'marker': '<', 'linestyle': '--'},
                 'mass premult 2': {'marker': '^', 'linestyle': '--'},
                 'mass premult 3': {'marker': 'v', 'linestyle': '--'},
                 'laplace premult 0': {'marker': '>', 'linestyle': '--'},
                 'laplace premult 1': {'marker': '<', 'linestyle': '--'},
                 'laplace premult 2': {'marker': '^', 'linestyle': '--'},
                 'laplace premult 3': {'marker': 'v', 'linestyle': '--'},
                 'helmholtz premult 0': {'marker': '>', 'linestyle': '--'},
                 'helmholtz premult 1': {'marker': '<', 'linestyle': '--'},
                 'helmholtz premult 2': {'marker': '^', 'linestyle': '--'},
                 'helmholtz premult 3': {'marker': 'v', 'linestyle': '--'}}

    def assembly(self, size=32, degree=1, dim=2, fs='scalar'):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()
        with self.timed_region('setup'):
            FS = {'scalar': FunctionSpace, 'vector': VectorFunctionSpace}[fs]
            V = FS(mesh, 'CG', degree)
            Q = FunctionSpace(mesh, 'CG', degree)

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            mass = inner(u, v)
            laplace = inner(grad(u), grad(v))
            f = Function(Q)
            g = Function(Q)
            h = Function(Q)
            f.interpolate(Expression('1.0'))
            g.interpolate(Expression('1.0'))
            h.interpolate(Expression('1.0'))
            A = assemble(mass*dx)

        with self.timed_region('mass premult 0'):
            assemble(mass*dx, tensor=A)
        with self.timed_region('mass premult 1'):
            assemble(f*mass*dx, tensor=A)
        with self.timed_region('mass premult 2'):
            assemble(g*f*mass*dx, tensor=A)
        with self.timed_region('mass premult 3'):
            assemble(h*g*f*mass*dx, tensor=A)
        with self.timed_region('laplace premult 0'):
            assemble(laplace*dx, tensor=A)
        with self.timed_region('laplace premult 1'):
            assemble(f*laplace*dx, tensor=A)
        with self.timed_region('laplace premult 2'):
            assemble(g*f*laplace*dx, tensor=A)
        with self.timed_region('laplace premult 3'):
            assemble(h*g*f*laplace*dx, tensor=A)
        with self.timed_region('helmholtz premult 0'):
            assemble((mass+laplace)*dx, tensor=A)
        with self.timed_region('helmholtz premult 1'):
            assemble(f*(mass+laplace)*dx, tensor=A)
        with self.timed_region('helmholtz premult 2'):
            assemble(g*f*(mass+laplace)*dx, tensor=A)
        with self.timed_region('helmholtz premult 3'):
            assemble(h*g*f*(mass+laplace)*dx, tensor=A)
        t = timings(True)
        for task in ['Assemble cells', 'Build sparsity']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    DolfinAssembly().main(benchmark=True, save=None)

    # Profile
    from itertools import product
    r0 = ['mass', 'laplace', 'helmholtz']
    r1 = ['assembly', 'assembly premult 1', 'assembly premult 2', 'assembly premult 3']
    regions = map(' '.join, product(r0, r1))
    DolfinAssembly().profile(regions=regions)
