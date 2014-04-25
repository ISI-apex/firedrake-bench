from navier_stokes import NavierStokes
from firedrake import *


class FiredrakeNavierStokes(NavierStokes):

    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'color': 'black',
                           'marker': '*',
                           'linestyle': '-'},
                 'mesh': {'color': 'blue',
                          'marker': '+',
                          'linestyle': '-'},
                 'setup': {'color': 'green',
                           'marker': 'x',
                           'linestyle': '-'},
                 'matrix assembly': {'color': 'blue',
                                     'marker': 'p',
                                     'linestyle': '-'},
                 'timestepping': {'color': 'yellow',
                                  'marker': 'o',
                                  'linestyle': '-'},
                 'tentative velocity RHS': {'color': 'magenta',
                                            'marker': '^',
                                            'linestyle': '-'},
                 'tentative velocity solve': {'color': 'magenta',
                                              'marker': 'v',
                                              'linestyle': '-'},
                 'pressure correction RHS': {'color': 'red',
                                             'marker': 's',
                                             'linestyle': '-'},
                 'pressure correction solve': {'color': 'red',
                                               'marker': 'D',
                                               'linestyle': '-'},
                 'velocity correction RHS': {'color': 'cyan',
                                             'marker': '>',
                                             'linestyle': '-'},
                 'velocity correction solve': {'color': 'cyan',
                                               'marker': '<',
                                               'linestyle': '-'}}

    def navier_stokes(self, scale=1, T=1, preassemble=True, save=False):
        with self.timed_region('mesh'):
            # Load mesh from file
            mesh = Mesh("lshape_%s.msh" % scale)

        with self.timed_region('setup'):
            # Define function spaces (P2-P1)
            V = VectorFunctionSpace(mesh, "Lagrange", 2)
            Q = FunctionSpace(mesh, "Lagrange", 1)

            # Define trial and test functions
            u = TrialFunction(V)
            p = TrialFunction(Q)
            v = TestFunction(V)
            q = TestFunction(Q)

            # Set parameter values
            dt = 0.01
            nu = 0.01

            # Define time-dependent pressure boundary condition
            p_in = Constant(0.0)

            # Define boundary conditions
            noslip = DirichletBC(V, Constant((0.0, 0.0)), (1, 3, 4, 6))
            inflow = DirichletBC(Q, p_in, 5)
            outflow = DirichletBC(Q, 0, 2)
            bcu = [noslip]
            bcp = [inflow, outflow]

            # Create functions
            u0 = Function(V)
            u1 = Function(V)
            p1 = Function(Q)

            # Define coefficients
            k = Constant(dt)
            f = Constant((0, 0))

            # Tentative velocity step
            F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
                nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
            a1 = lhs(F1)
            L1 = rhs(F1)

            # Pressure update
            a2 = inner(grad(p), grad(q))*dx
            L2 = -(1/k)*div(u1)*q*dx

            # Velocity update
            a3 = inner(u, v)*dx
            L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

        if preassemble:
            with self.timed_region('matrix assembly'):
                # Assemble matrices
                A1 = assemble(a1)
                A2 = assemble(a2)
                A3 = assemble(a3)

        if save:
            # Create files for storing solution
            ufile = File("vtk/firedrake_velocity.pvd")
            pfile = File("vtk/firedrake_pressure.pvd")

        vparams = {'ksp_type': 'gmres',
                   'pc_type': 'ilu',
                   'ksp_rtol': 1e-6,
                   'ksp_atol': 1e-15}
        pparams = {'ksp_type': 'cg',
                   'pc_type': 'ilu',
                   'ksp_rtol': 1e-6,
                   'ksp_atol': 1e-15}

        with self.timed_region('timestepping'):
            # Time-stepping
            t = dt
            while t < T + 1e-14:
                # Update pressure boundary condition
                p_in.assign(sin(3.0*t))

                # Compute tentative velocity step
                info("Computing tentative velocity")
                if preassemble:
                    with self.timed_region('tentative velocity RHS'):
                        b1 = assemble(L1)
                        [bc.apply(A1, b1) for bc in bcu]
                        b1.dat.data
                    with self.timed_region('tentative velocity solve'):
                        solve(A1, u1, b1, solver_parameters=vparams)
                else:
                    with self.timed_region('tentative velocity solve'):
                        solve(a1 == L1, u1, bcs=bcu, solver_parameters=vparams)

                # Pressure correction
                info("Computing pressure correction")
                if preassemble:
                    with self.timed_region('pressure correction RHS'):
                        b2 = assemble(L2)
                        [bc.apply(A2, b2) for bc in bcp]
                        b2.dat.data
                    with self.timed_region('pressure correction solve'):
                        solve(A2, p1, b2, solver_parameters=pparams)
                else:
                    with self.timed_region('pressure correction solve'):
                        solve(a2 == L2, p1, bcs=bcp, solver_parameters=pparams)

                # Velocity correction
                info("Computing velocity correction")
                if preassemble:
                    with self.timed_region('velocity correction RHS'):
                        b3 = assemble(L3)
                        [bc.apply(A3, b3) for bc in bcu]
                        b3.dat.data
                    with self.timed_region('velocity correction solve'):
                        solve(A3, u1, b3, solver_parameters=vparams)
                else:
                    with self.timed_region('velocity correction solve'):
                        solve(a3 == L3, u1, bcs=bcu, solver_parameters=vparams)

                if save:
                    # Save to file
                    ufile << u1
                    pfile << p1

                # Move to next time step
                u0.assign(u1)
                t += dt

if __name__ == '__main__':
    op2.init(log_level='WARNING')

    # Benchmark
    FiredrakeNavierStokes().main(benchmark=True, save=None)

    # Output VTU files
    # FiredrakeNavierStokes().navier_stokes(save=True)

    # Profile
    # from itertools import product
    # r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
    # r2 = ['RHS', 'solve']
    # regions = ['matrix assembly'] + map(' '.join, product(r1, r2))
    # FiredrakeNavierStokes().profile(regions=regions)
