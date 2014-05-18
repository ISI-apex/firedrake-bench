from mixed_forms import MixedForms
from firedrake import *
# from pyop2.ir.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers

parameters["assembly_cache"]["enabled"] = False
parameters["coffee"]["licm"] = True
# Vectorization appears to degrade performance under some circumstances
# parameters["coffee"]["ap"] = True
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


def poisson(degree, qdegree, dim, mesh):
    BDM = FunctionSpace(mesh, "BDM", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    return dot(sigma, tau) + div(tau)*u + div(sigma)*v


class FiredrakeMixedForms(MixedForms):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'nf 0': {'marker': '>', 'linestyle': '-'},
                 'nf 1': {'marker': '<', 'linestyle': '-'},
                 'nf 2': {'marker': '^', 'linestyle': '-'}}

    def mixed_forms(self, degree=1, qdegree=1, dim=2, form='mass'):
        mesh = UnitSquareMesh(31, 31) if dim == 2 else UnitCubeMesh(9, 9, 9)
        it = eval(form)(degree, qdegree, dim, mesh)
        Q = FunctionSpace(mesh, 'CG', qdegree)
        f = [Function(Q).assign(1.0) for _ in range(3)]
        A = assemble(it*dx)

        for nf in range(3):
            with self.timed_region('nf %d' % nf):
                assemble(reduce(inner, f[:nf] + [it])*dx, tensor=A)
                A.M
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeMixedForms().main()
