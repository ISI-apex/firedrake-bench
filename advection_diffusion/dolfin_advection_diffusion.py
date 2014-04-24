from advection_diffusion import AdvectionDiffusion
from dolfin import *

parameters["reorder_dofs_serial"] = False

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


class DolfinAdvectionDiffusion(AdvectionDiffusion):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'color': 'black',
                           'marker': '*',
                           'linestyle': '--'},
                 'mesh': {'color': 'blue',
                          'marker': '+',
                          'linestyle': '--'},
                 'setup': {'color': 'green',
                           'marker': 'x',
                           'linestyle': '--'},
                 'advection matrix': {'color': 'cyan',
                                      'marker': '>',
                                      'linestyle': '--'},
                 'diffusion matrix': {'color': 'cyan',
                                      'marker': '<',
                                      'linestyle': '--'},
                 'timestepping': {'color': 'yellow',
                                  'marker': 'o',
                                  'linestyle': '--'},
                 'advection RHS': {'color': 'magenta',
                                   'marker': '^',
                                   'linestyle': '--'},
                 'diffusion RHS': {'color': 'magenta',
                                   'marker': 'v',
                                   'linestyle': '--'},
                 'advection solve': {'color': 'red',
                                     'marker': 's',
                                     'linestyle': '--'},
                 'diffusion solve': {'color': 'red',
                                     'marker': 'D',
                                     'linestyle': '--'}}

    def advection_diffusion(self, size=32, degree=1, dim=2, dt=0.0001, T=0.01,
                            diffusivity=0.1, advection=True, diffusion=True):
        with self.timed_region('mesh'):
            mesh = make_mesh[dim](size)
            mesh.init()

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
            while T < 0.02:

                # Advection
                if advection:
                    with self.timed_region('advection RHS'):
                        b = assemble(adv_rhs)
                    with self.timed_region('advection solve'):
                        solve(A, t.vector(), b, "cg", "jacobi")

                # Diffusion
                if diffusion:
                    with self.timed_region('diffusion RHS'):
                        b = assemble(diff_rhs)
                    with self.timed_region('diffusion solve'):
                        solve(D, t.vector(), b, "cg", "jacobi")

                T = T + dt

        # Analytical solution
        a = Function(V)
        a.interpolate(Expression(fexpr % {'T': T}))
        l2 = sqrt(assemble(dot(t - a, t - a) * dx))
        if MPI.rank(mpi_comm_world()) == 0:
            print 'L2 error norm:', l2

if __name__ == '__main__':
    set_log_active(False)
    DolfinAdvectionDiffusion().main(benchmark=True, save=None)
    # Profile
    # DolfinAdvectionDiffusion().profile(regions=['advection RHS', 'advection solve', 'diffusion RHS', 'diffusion solve'])
