from forms import Forms
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


def mass(degree, qdegree, dim, mesh):
    V = FunctionSpace(mesh, 'CG', degree)
    Q = FunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = dot(v, u)
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


def elasticity(degree, qdegree, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    Q = FunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda v: grad(v) + transpose(grad(v))
    it = 0.25*inner(eps(v), eps(u))
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


def poisson(degree, qdegree, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    Q = VectorFunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = inner(grad(v), grad(u))
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression(('1.0',)*dim))
    return it, f, div


def mixed_poisson(degree, qdegree, dim, mesh):
    BDM = FunctionSpace(mesh, "BDM", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    Q = FunctionSpace(mesh, 'CG', qdegree)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    it = dot(sigma, tau) + div(tau)*u + div(sigma)*v
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


class DolfinForms(Forms):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'nf 0': {'marker': '>', 'linestyle': '--'},
                 'nf 1': {'marker': '<', 'linestyle': '--'},
                 'nf 2': {'marker': '^', 'linestyle': '--'},
                 'nf 3': {'marker': 'v', 'linestyle': '--'}}

    def forms(self, degree=1, qdegree=1, dim=2, form='mass'):
        mesh = UnitSquareMesh(31, 31) if dim == 2 else UnitCubeMesh(9, 9, 9)
        it, f, m = eval(form)(degree, qdegree, dim, mesh)
        A = assemble(it*dx)

        for nf in range(4):
            with self.timed_region('nf %d' % nf):
                assemble(reduce(inner, map(m, f[:nf]) + [it])*dx, tensor=A)
        t = timings(True)
        task = 'Assemble cells'
        self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinForms().main()
