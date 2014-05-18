from mixed_forms import MixedForms
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


def poisson(degree, qdegree, dim, mesh):
    BDM = FunctionSpace(mesh, "BDM", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    return dot(sigma, tau) + div(tau)*u + div(sigma)*v


class DolfinMixedForms(MixedForms):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'nf 0': {'marker': '>', 'linestyle': '--'},
                 'nf 1': {'marker': '<', 'linestyle': '--'},
                 'nf 2': {'marker': '^', 'linestyle': '--'}}

    def mixed_forms(self, degree=1, qdegree=1, dim=2, form='poisson'):
        mesh = UnitSquareMesh(31, 31) if dim == 2 else UnitCubeMesh(9, 9, 9)
        it = eval(form)(degree, qdegree, dim, mesh)
        Q = FunctionSpace(mesh, 'CG', qdegree)
        f = [Function(Q) for _ in range(3)]
        for f_ in f:
            f_.interpolate(Expression('1.0'))
        A = assemble(it*dx)

        for nf in range(3):
            with self.timed_region('nf %d' % nf):
                assemble(reduce(inner, f[:nf] + [it])*dx, tensor=A)
        t = timings(True)
        task = 'Assemble cells'
        self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinMixedForms().main()
