from forms import Forms
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


def mass(p, q, dim, mesh):
    V = FunctionSpace(mesh, 'CG', p)
    Q = FunctionSpace(mesh, 'CG', q)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = dot(v, u)
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


def elasticity(p, q, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', p)
    Q = FunctionSpace(mesh, 'CG', q)
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda v: grad(v) + transpose(grad(v))
    it = 0.25*inner(eps(v), eps(u))
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


def poisson(p, q, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', p)
    Q = VectorFunctionSpace(mesh, 'CG', q)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = inner(grad(v), grad(u))
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression(('1.0',)*dim))
    return it, f, div


def mixed_poisson(p, q, dim, mesh):
    BDM = FunctionSpace(mesh, "BDM", p)
    DG = FunctionSpace(mesh, "DG", p - 1)
    Q = FunctionSpace(mesh, 'CG', q)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    it = dot(sigma, tau) + div(tau)*u + div(sigma)*v
    f = [Function(Q) for _ in range(3)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return it, f, lambda x: x


class DolfinForms(Forms):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}

    def forms(self, p=1, q=1, dim=3, form='mass'):
        if dim == 2:
            mesh = UnitSquareMesh(31, 31)
            normalize = 1.0
        if dim == 3:
            size = int(18.0 / (p+q))
            normalize = 1000.0 / (size+1)**3
            mesh = UnitCubeMesh(size, size, size)
        it, f, m = eval(form)(p, q, dim, mesh)
        A = assemble(it*dx)

        for nf in range(4):
            with self.timed_region('nf %d' % nf, normalize):
                assemble(reduce(inner, map(m, f[:nf]) + [it])*dx, tensor=A)
        t = timings(True)
        task = 'Assemble cells'
        self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinForms().main()
