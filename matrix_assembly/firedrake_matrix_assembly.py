from matrix_assembly import MatrixAssembly
from firedrake import *
from pyop2.ir.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

parameters["assembly_cache"]["max_factor"] = float("inf")
parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


class FiredrakeMatrixAssembly(MatrixAssembly):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'mesh': {'marker': '+', 'linestyle': '-'},
                 'setup': {'marker': 'x', 'linestyle': '-'},
                 'assembly': {'marker': '>', 'linestyle': '-'},
                 'reassembly': {'marker': '<', 'linestyle': '-'},
                 'assembly bcs': {'marker': '^', 'linestyle': '-'},
                 'reassembly bcs': {'marker': 'v', 'linestyle': '-'}}

    def matrix_assembly(self, size=32, degree=1, dim=2, fs='scalar'):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
        with self.timed_region('setup'):
            FS = {'scalar': FunctionSpace, 'vector': VectorFunctionSpace}[fs]
            V = FS(mesh, "Lagrange", degree)

            # Define boundary condition
            bc = DirichletBC(V, 0.0, [3, 4])

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            a = inner(grad(u), grad(v))*dx

        with self.timed_region('assembly'):
            A = assemble(a)
            A.M
        with self.timed_region('reassembly'):
            A = assemble(a, tensor=A)
            A.M
        # Clear sparsity cache
        A.M.sparsity.dsets[0].set._cache.clear()
        with self.timed_region('assembly bcs'):
            A = assemble(a, bcs=bc)
            A.M
        with self.timed_region('reassembly bcs'):
            A = assemble(a, tensor=A, bcs=bc)
            A.M
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')

    FiredrakeMatrixAssembly().main()
