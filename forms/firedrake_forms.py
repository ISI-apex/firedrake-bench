from forms import Forms
from firedrake import *
# from pyop2.ir.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers

parameters["assembly_cache"]["enabled"] = False
parameters["coffee"]["licm"] = True
# Vectorization appears to degrade performance under some circumstances
# parameters["coffee"]["ap"] = True
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


def mass(degree, qdegree, dim, mesh):
    V = FunctionSpace(mesh, 'CG', degree)
    Q = FunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = dot(v, u)
    f = [Function(Q).assign(1.0) for _ in range(3)]
    return it, f, lambda x: x


def elasticity(degree, qdegree, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    Q = FunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    eps = lambda v: grad(v) + transpose(grad(v))
    it = 0.25*inner(eps(v), eps(u))
    f = [Function(Q).assign(1.0) for _ in range(3)]
    return it, f, lambda x: x


def poisson(degree, qdegree, dim, mesh):
    V = VectorFunctionSpace(mesh, 'CG', degree)
    Q = VectorFunctionSpace(mesh, 'CG', qdegree)
    u = TrialFunction(V)
    v = TestFunction(V)
    it = inner(grad(v), grad(u))
    f = [Function(Q).assign(1.0) for _ in range(3)]
    return it, f, div


class FiredrakeForms(Forms):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'nf 0': {'marker': '>', 'linestyle': '-'},
                 'nf 1': {'marker': '<', 'linestyle': '-'},
                 'nf 2': {'marker': '^', 'linestyle': '-'},
                 'nf 3': {'marker': 'v', 'linestyle': '-'}}

    def forms(self, degree=1, qdegree=1, dim=2, form='mass'):
        mesh = UnitSquareMesh(31, 31) if dim == 2 else UnitCubeMesh(9, 9, 9)
        it, f, m = eval(form)(degree, qdegree, dim, mesh)
        A = assemble(it*dx)

        for nf in range(4):
            with self.timed_region('nf %d' % nf):
                assemble(reduce(inner, map(m, f[:nf]) + [it])*dx, tensor=A)
                A.M
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeForms().main()
