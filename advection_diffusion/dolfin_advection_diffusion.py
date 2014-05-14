from advection_diffusion import AdvectionDiffusion
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinAdvectionDiffusion(AdvectionDiffusion):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'mesh': {'marker': '+', 'linestyle': '--'},
                 'setup': {'marker': 'x', 'linestyle': '--'},
                 'advection matrix': {'marker': '>', 'linestyle': '--'},
                 'diffusion matrix': {'marker': '<', 'linestyle': '--'},
                 'timestepping': {'marker': 'o', 'linestyle': '--'},
                 'advection RHS': {'marker': '^', 'linestyle': '--'},
                 'diffusion RHS': {'marker': 'v', 'linestyle': '--'},
                 'advection solve': {'marker': 's', 'linestyle': '--'},
                 'diffusion solve': {'marker': 'D', 'linestyle': '--'}}

    def advection_diffusion(self, scale=1.0, mesh='square', degree=1, dim=2,
                            dt=0.0001, T=0.01, diffusivity=0.1,
                            advection=True, diffusion=True,
                            print_norm=False, pc='amg'):
        with self.timed_region('mesh'):
            mesh = Mesh("meshes/%s_%s.xml.gz" % (mesh, scale))

        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "CG", degree)
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
            u.interpolate(Expression(('1.0', '0.0')))

        if advection:
            with self.timed_region('advection matrix'):
                A = assemble(adv)
        if diffusion:
            with self.timed_region('diffusion matrix'):
                D = assemble(diff)

        with self.timed_region('timestepping'):
            while T < 0.011:

                # Advection
                if advection:
                    with self.timed_region('advection RHS'):
                        b = assemble(adv_rhs)
                    with self.timed_region('advection solve'):
                        solve(A, t.vector(), b, "cg", pc)

                # Diffusion
                if diffusion:
                    with self.timed_region('diffusion RHS'):
                        b = assemble(diff_rhs)
                    with self.timed_region('diffusion solve'):
                        solve(D, t.vector(), b, "cg", pc)

                T = T + dt

        # Analytical solution
        a = Function(V)
        a.interpolate(Expression(fexpr % {'T': T}))
        l2 = sqrt(assemble(dot(t - a, t - a) * dx))
        if print_norm and MPI.rank(mpi_comm_world()) == 0:
            print 'L2 error norm:', l2
        t = timings(True)
        for task in ['Assemble cells', 'Build sparsity', 'PETSc Krylov solver']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)

    # Benchmark
    DolfinAdvectionDiffusion().main(benchmark=True, save=None)

    # Profile
    regions = ['advection RHS', 'advection solve', 'diffusion RHS', 'diffusion solve']
    DolfinAdvectionDiffusion().profile(regions=regions)
