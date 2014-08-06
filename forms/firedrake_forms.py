from forms import Forms
from firedrake import *
from firedrake import __version__ as firedrake_version
# from pyop2.coffee.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers
from pyop2 import __version__ as pyop2_version

parameters["assembly_cache"]["enabled"] = False
parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
# Vectorisation caused slowdowns for some forms
# parameters["coffee"]["vect"] = (V_OP_UAJ, 1)

meshes = {2: UnitSquareMesh(31, 31), 3: UnitCubeMesh(9, 9, 9)}


def mass(q, p, dim, mesh, nf=0):
    V = FunctionSpace(mesh, 'CG', q)
    P = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = dot(v, u)
    f = [Function(P).assign(1.0) for _ in range(nf)]
    return reduce(inner, f + [it])*dx


def elasticity(q, p, dim, mesh, nf=0):
    V = VectorFunctionSpace(mesh, 'CG', q)
    P = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda v: grad(v) + transpose(grad(v))
    it = 0.25*inner(eps(v), eps(u))
    f = [Function(P).assign(1.0) for _ in range(nf)]
    return reduce(inner, f + [it])*dx


def poisson(q, p, dim, mesh, nf=0):
    V = VectorFunctionSpace(mesh, 'CG', q)
    P = VectorFunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = inner(grad(v), grad(u))
    f = [div(Function(P).assign(1.0)) for _ in range(nf)]
    return reduce(inner, f + [it])*dx


def mixed_poisson(q, p, dim, mesh, nf=0):
    BDM = FunctionSpace(mesh, "BDM", q)
    DG = FunctionSpace(mesh, "DG", q - 1)
    P = FunctionSpace(mesh, 'CG', p)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    it = dot(sigma, tau) + div(tau)*u + div(sigma)*v
    f = [Function(P).assign(1.0) for _ in range(nf)]
    return reduce(inner, f + [it])*dx


class FiredrakeForms(Forms):
    series = {'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"],
            'firedrake': firedrake_version,
            'pyop2': pyop2_version}

    def forms(self, q=1, p=1, dim=3, max_nf=3, form='mass'):
        mesh = meshes[dim]
        A = assemble(eval(form)(q, p, dim, mesh))

        for nf in range(max_nf + 1):
            f = eval(form)(q, p, dim, mesh, nf)
            with self.timed_region('nf %d' % nf):
                assemble(f, tensor=A)
                A.M
        t = get_timers(reset=True)
        task = 'Assemble cells'
        self.register_timing(task, t[task].total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeForms().main()
