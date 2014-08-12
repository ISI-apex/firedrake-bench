from advection_diffusion import AdvectionDiffusion
from pybench import timed
from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
# from pyop2.coffee.ast_plan import V_OP_UAJ
from pyop2.profiling import get_timers
from pyop2 import __version__ as pyop2_version

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
# Vectorization appears to degrade performance for p2
# parameters["coffee"]["vect"] = (V_OP_UAJ, 3)


class FiredrakeAdvectionDiffusion(AdvectionDiffusion):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"],
            'firedrake': firedrake_version,
            'pyop2': pyop2_version}

    @memoize
    @timed
    def make_mesh(self, dim, x):
        return UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)

    def advection_diffusion(self, size=64, degree=1, dim=2, verbose=False,
                            dt=0.0001, T=0.01, Tend=0.011, diffusivity=0.1,
                            advection=True, diffusion=True, weak=False,
                            print_norm=False, preassemble=True, pc='hypre',
                            strong_threshold=0.75, agg_nl=2, max_levels=25):
        if weak:
            size = int((1e4*op2.MPI.comm.size)**(1./dim))
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.meta['cells'] = (2 if dim == 2 else 6)*size**dim
        self.meta['dofs'] = (size+1)**dim
        solver_parameters = {'ksp_type': 'cg',
                             'pc_type': pc,
                             'pc_hypre_type': 'boomeramg',
                             'pc_hypre_boomeramg_strong_threshold': strong_threshold,
                             'pc_hypre_boomeramg_agg_nl': agg_nl,
                             'pc_hypre_boomeramg_max_levels': max_levels,
                             'ksp_rtol': 1e-6,
                             'ksp_atol': 1e-15}
        if verbose:
            solver_parameters['pc_hypre_boomeramg_print_statistics'] = True
            solver_parameters['ksp_view'] = True
            solver_parameters['ksp_monitor'] = True
        t_, mesh = self.make_mesh(dim, size)
        self.register_timing('mesh', t_)

        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "CG", degree)
            if verbose:
                print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size
            U = VectorFunctionSpace(mesh, "CG", degree)

            p = TrialFunction(V)
            q = TestFunction(V)
            t = Function(V)
            u = Function(U)

            adv = p * q * dx
            adv_rhs = (q * t + dt * dot(grad(q), u) * t) * dx

            d = -dt * diffusivity * dot(grad(q), grad(p)) * dx

            diff = adv - 0.5 * d
            diff_rhs = action(adv + 0.5 * d, t)

            # Set initial condition:
            # A*(e^(-r^2/(4*D*T)) / (4*pi*D*T))
            # with normalisation A = 0.1, diffusivity D = 0.1
            r2 = "(pow(x[0]-(0.45+%(T)f), 2.0) + pow(x[1]-0.5, 2.0))"
            fexpr = "0.1 * (exp(-" + r2 + "/(0.4*%(T)f)) / (0.4*pi*%(T)f))"
            t.interpolate(Expression(fexpr % {'T': T}))
            u.interpolate(Expression((1.0, 0.0)))
            t.dat._force_evaluation()
            u.dat._force_evaluation()

        if preassemble:
            if advection:
                with self.timed_region('advection matrix'):
                    A = assemble(adv)
                    A.M
            if diffusion:
                with self.timed_region('diffusion matrix'):
                    D = assemble(diff)
                    D.M

        with self.timed_region('timestepping'):
            while T < Tend:

                # Advection
                if advection:
                    if preassemble:
                        with self.timed_region('advection RHS'):
                            b = assemble(adv_rhs)
                            b.dat._force_evaluation()
                        with self.timed_region('advection solve'):
                            solve(A, t, b, solver_parameters=solver_parameters)
                    else:
                        solve(adv == adv_rhs, t, solver_parameters=solver_parameters)

                # Diffusion
                if diffusion:
                    if preassemble:
                        with self.timed_region('diffusion RHS'):
                            b = assemble(diff_rhs)
                            b.dat._force_evaluation()
                        with self.timed_region('diffusion solve'):
                            solve(D, t, b, solver_parameters=solver_parameters)
                    else:
                        solve(diff == diff_rhs, t, solver_parameters=solver_parameters)

                T = T + dt

        # Analytical solution
        a = Function(V).interpolate(Expression(fexpr % {'T': T}))
        l2 = sqrt(assemble(dot(t - a, t - a) * dx))
        if print_norm and op2.MPI.comm.rank == 0:
            print 'L2 error norm:', l2
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeAdvectionDiffusion().main()
