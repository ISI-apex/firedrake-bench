from assembly import Assembly
from firedrake import *
# from pyop2.ir.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

parameters["assembly_cache"]["enabled"] = False
parameters["coffee"]["licm"] = True
# Vectorization appears to degrade performance with gcc
# parameters["coffee"]["ap"] = True
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


class FiredrakeAssembly(Assembly):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'mesh': {'marker': '+', 'linestyle': '-'},
                 'setup': {'marker': 'x', 'linestyle': '-'},
                 'mass assembly': {'marker': '>', 'linestyle': '--'},
                 'mass assembly premult 1': {'marker': '<', 'linestyle': '-'},
                 'mass assembly premult 2': {'marker': '^', 'linestyle': '-'},
                 'mass assembly premult 3': {'marker': 'v', 'linestyle': '-'},
                 'laplace assembly': {'marker': '1', 'linestyle': '-'},
                 'laplace assembly premult 1': {'marker': '2', 'linestyle': '-'},
                 'laplace assembly premult 2': {'marker': '3', 'linestyle': '-'},
                 'laplace assembly premult 3': {'marker': '4', 'linestyle': '-'},
                 'helmholtz assembly': {'marker': 's', 'linestyle': '-'},
                 'helmholtz assembly premult 1': {'marker': 'p', 'linestyle': '-'},
                 'helmholtz assembly premult 2': {'marker': 'h', 'linestyle': '-'},
                 'helmholtz assembly premult 3': {'marker': 'D', 'linestyle': '-'}}

    def assembly(self, size=32, degree=1, dim=2, fs='scalar'):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
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
            A = assemble(mass*dx)

        with self.timed_region('mass assembly'):
            assemble(mass*dx, tensor=A)
            A.M
        with self.timed_region('mass assembly premult 1'):
            assemble(f*mass*dx, tensor=A)
            A.M
        with self.timed_region('mass assembly premult 2'):
            assemble(g*f*mass*dx, tensor=A)
            A.M
        with self.timed_region('mass assembly premult 3'):
            assemble(h*g*f*mass*dx, tensor=A)
            A.M
        with self.timed_region('laplace assembly'):
            assemble(laplace*dx, tensor=A)
            A.M
        with self.timed_region('laplace assembly premult 1'):
            assemble(f*laplace*dx, tensor=A)
            A.M
        with self.timed_region('laplace assembly premult 2'):
            assemble(g*f*laplace*dx, tensor=A)
            A.M
        with self.timed_region('laplace assembly premult 3'):
            assemble(h*g*f*laplace*dx, tensor=A)
            A.M
        with self.timed_region('helmholtz assembly'):
            assemble((mass+laplace)*dx, tensor=A)
            A.M
        with self.timed_region('helmholtz assembly premult 1'):
            assemble(f*(mass+laplace)*dx, tensor=A)
            A.M
        with self.timed_region('helmholtz assembly premult 2'):
            assemble(g*f*(mass+laplace)*dx, tensor=A)
            A.M
        with self.timed_region('helmholtz assembly premult 3'):
            assemble(h*g*f*(mass+laplace)*dx, tensor=A)
            A.M
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')

    # Benchmark
    FiredrakeAssembly().main(benchmark=True, save=None)

    # Profile
    from itertools import product
    r0 = ['mass', 'laplace', 'helmholtz']
    r1 = ['assembly', 'assembly premult 1', 'assembly premult 2', 'assembly premult 3']
    regions = map(' '.join, product(r0, r1))
    FiredrakeAssembly().profile(regions=regions)
