from advection_diffusion import AdvectionDiffusion
from firedrake import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}
solver_parameters = {'ksp_type': 'cg',
                     'pc_type': 'jacobi',
                     'ksp_rtol': 1e-6,
                     'ksp_atol': 1e-15}

class FiredrakeAdvectionDiffusion(AdvectionDiffusion):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'mesh': {'marker': '+', 'linestyle': '-'},
                 'setup': {'marker': 'x', 'linestyle': '-'},
                 'advection matrix': {'marker': '>', 'linestyle': '-'},
                 'diffusion matrix': {'marker': '<', 'linestyle': '-'},
                 'timestepping': {'marker': 'o', 'linestyle': '-'},
                 'advection RHS': {'marker': '^', 'linestyle': '-'},
                 'diffusion RHS': {'marker': 'v', 'linestyle': '-'},
                 'advection solve': {'marker': 's', 'linestyle': '-'},
                 'diffusion solve': {'marker': 'D', 'linestyle': '-'}}

    def advection_diffusion(self, size=32, degree=1, dim=2, dt=0.0001, T=0.01,
                            diffusivity=0.1, advection=True, diffusion=True,
                            print_norm=False):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)

        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "CG", degree)
            U = VectorFunctionSpace(mesh, "CG", degree)

            p = TrialFunction(V)
            q = TestFunction(V)
            t = Function(V)
            u = Function(U)

            diffusivity = 0.1

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

        if advection:
            with self.timed_region('advection matrix'):
                A = assemble(adv)
                A.M
        if diffusion:
            with self.timed_region('diffusion matrix'):
                D = assemble(diff)
                D.M

        with self.timed_region('timestepping'):
            while T < 0.02:

                # Advection
                if advection:
                    with self.timed_region('advection RHS'):
                        b = assemble(adv_rhs)
                        b.dat.data
                    with self.timed_region('advection solve'):
                        solve(A, t, b, solver_parameters=solver_parameters)

                # Diffusion
                if diffusion:
                    with self.timed_region('diffusion RHS'):
                        b = assemble(diff_rhs)
                        b.dat.data
                    with self.timed_region('diffusion solve'):
                        solve(D, t, b, solver_parameters=solver_parameters)

                T = T + dt

        # Analytical solution
        a = Function(V).interpolate(Expression(fexpr % {'T': T}))
        l2 = sqrt(assemble(dot(t - a, t - a) * dx))
        if print_norm and op2.MPI.comm.rank == 0:
            print 'L2 error norm:', l2

if __name__ == '__main__':
    op2.init(log_level='WARNING')

    # Benchmark
    FiredrakeAdvectionDiffusion().main(benchmark=True, save=None)

    # Profile
    regions = ['advection RHS', 'advection solve', 'diffusion RHS', 'diffusion solve']
    FiredrakeAdvectionDiffusion().profile(regions=regions)
