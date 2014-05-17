from forms import Forms
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinForms(Forms):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'nf 0': {'marker': '>', 'linestyle': '--'},
                 'nf 1': {'marker': '<', 'linestyle': '--'},
                 'nf 2': {'marker': '^', 'linestyle': '--'},
                 'nf 3': {'marker': 'v', 'linestyle': '--'}}

    def forms(self, degree=1, qdegree=1, dim=2, form='mass'):
        mesh = UnitSquareMesh(31, 31) if dim == 2 else UnitCubeMesh(9, 9, 9)
        FS = {'mass': FunctionSpace,
              'elasticity': VectorFunctionSpace}[form]
        V = FS(mesh, 'CG', degree)
        Q = FunctionSpace(mesh, 'CG', qdegree)

        u = TrialFunction(V)
        v = TestFunction(V)

        if form == 'mass':
            it = dot(v, u)
        if form == 'elasticity':
            eps = lambda v: grad(v) + transpose(grad(v))
            it = 0.25*inner(eps(v), eps(u))
        f = [Function(Q) for _ in range(3)]
        for f_ in f:
            f_.interpolate(Expression('1.0'))
        A = assemble(it*dx)

        for nf in range(4):
            with self.timed_region('nf %d' % nf):
                assemble(reduce(inner, f[:nf] + [it])*dx, tensor=A)
        t = timings(True)
        task = 'Assemble cells'
        self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinForms().main()
