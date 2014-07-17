from forms import Forms
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

meshes = {2: UnitSquareMesh(31, 31), 3: UnitCubeMesh(9, 9, 9)}


def mass(q, p, dim, mesh, nf=0):
    V = FunctionSpace(mesh, 'CG', q)
    P = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = dot(v, u)
    f = [Function(P) for _ in range(nf)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return reduce(inner, f + [it])*dx


def elasticity(q, p, dim, mesh, nf=0):
    V = VectorFunctionSpace(mesh, 'CG', q)
    P = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda v: grad(v) + transpose(grad(v))
    it = 0.25*inner(eps(v), eps(u))
    f = [Function(P) for _ in range(nf)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return reduce(inner, f + [it])*dx


def poisson(q, p, dim, mesh, nf=0):
    V = VectorFunctionSpace(mesh, 'CG', q)
    P = VectorFunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = inner(grad(v), grad(u))
    f = [Function(P) for _ in range(nf)]
    for f_ in f:
        f_.interpolate(Expression(('1.0',)*dim))
    return reduce(inner, map(div, f) + [it])*dx


def mixed_poisson(q, p, dim, mesh, nf=0):
    BDM = FunctionSpace(mesh, "BDM", q)
    DG = FunctionSpace(mesh, "DG", q - 1)
    P = FunctionSpace(mesh, 'CG', p)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    it = dot(sigma, tau) + div(tau)*u + div(sigma)*v
    f = [Function(P) for _ in range(nf)]
    for f_ in f:
        f_.interpolate(Expression('1.0'))
    return reduce(inner, f + [it])*dx


class DolfinForms(Forms):
    series = {'variant': 'DOLFIN'}

    def forms(self, q=1, p=1, dim=3, form='mass'):
        mesh = meshes[dim]
        A = assemble(eval(form)(q, p, dim, mesh))

        for nf in range(4):
            f = eval(form)(q, p, dim, mesh, nf)
            with self.timed_region('nf %d' % nf):
                assemble(f, tensor=A)
        t = timings(True)
        task = 'Assemble cells'
        self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinForms().main()
