from elasticity import Elasticity
from pybench import timed
from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
from pyop2.profiling import get_timers
from pyop2 import __version__ as pyop2_version
from pyop2.coffee.ast_plan import V_OP_UAJ


#parameters["coffee"]["licm"] = True
#parameters["coffee"]["ap"] = True
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)
#parameters["coffee"]["autotune"] = True


class FiredrakeElasticity(Elasticity):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"],
            'firedrake': firedrake_version,
            'pyop2': pyop2_version}

    @memoize
    @timed
    def make_mesh(self, dim, x):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)

    def elasticity(self, size=32, degree=1, dim=3, preassemble=True, weak=False,
                   nf=0, verbose=False, opt=(0, False, 0)):
        parameters["coffee"]["licm"] = opt[0]
        parameters["coffee"]["ap"] = opt[1]
        parameters["coffee"]["split"] = opt[2]
        if weak:
            size = int((size*op2.MPI.comm.size)**(1./dim))
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.series['degree'] = degree
        self.meta['cells'] = 6*size**dim
        self.meta['vertices'] = (size+1)**dim
        t_, mesh = self.make_mesh(dim, size)
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = VectorFunctionSpace(mesh, "CG", degree)
            P = FunctionSpace(mesh, 'CG', degree)
            if verbose:
                print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size

            # Define boundary condition
            bc = DirichletBC(V, 0.0, [3, 4])

            # Define variational problem
            U = TrialFunction(V)
            v = TestFunction(V)
            f = Function(V)
            f.interpolate(Expression(("cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])",
                                      "cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])",
                                      "cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])")))
            f.dat._force_evaluation()
            nf = [Function(P).assign(2.0) for _ in range(nf)]

            E = 10.0
            nu = 0.3
            mu = E / (2*(1 + nu))
            lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

            def epsilon(v):
                return 0.5*(grad(v) + transpose(grad(v)))

            def sigma(v):
                return 2*mu*epsilon(v) + \
                    lmbda*tr(epsilon(v))*Identity(len(v))

            a = inner(grad(v), sigma(U))
            a = reduce(inner, nf + [a])*dx
            L = dot(v, f)*dx

            # Compute solution
            x = Function(V)
        if preassemble:
            with self.timed_region('matrix assembly'):
                A = assemble(a)
                A.M
                self.meta['dofs'] = A.M.handle.sizes[0][1]
            with self.timed_region('rhs assembly'):
                b = assemble(L)
                b.dat._force_evaluation()
            with self.timed_region('solve'):
                solve(A, x, b)
                x.dat._force_evaluation()
        else:
            with self.timed_region('solve'):
                solve(a == L, x, bcs=[bc])
                x.dat._force_evaluation()

        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeElasticity().main()
