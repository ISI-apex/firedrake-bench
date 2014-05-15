from navier_stokes import NavierStokes
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

PETScOptions.set("sub_pc_type", "ilu")


class DolfinNavierStokes(NavierStokes):

    series = {'np': MPI.size(mpi_comm_world())}
    plotstyle = {'total': {'marker': '*', 'linestyle': '--'},
                 'mesh': {'marker': '+', 'linestyle': '--'},
                 'setup': {'marker': 'x', 'linestyle': '--'},
                 'matrix assembly': {'marker': 'p', 'linestyle': '--'},
                 'timestepping': {'marker': 'o', 'linestyle': '--'},
                 'tentative velocity RHS': {'marker': '^', 'linestyle': '--'},
                 'tentative velocity solve': {'marker': 'v', 'linestyle': '--'},
                 'pressure correction RHS': {'marker': 's', 'linestyle': '--'},
                 'pressure correction solve': {'marker': 'D', 'linestyle': '--'},
                 'velocity correction RHS': {'marker': '>', 'linestyle': '--'},
                 'velocity correction solve': {'marker': '<', 'linestyle': '--'}}

    def navier_stokes(self, scale=1, T=1, preassemble=True, save=False,
                      compute_norms=False, symmetric=True):
        with self.timed_region('mesh'):
            # Load mesh from file
            mesh = Mesh("meshes/lshape_%s.xml.gz" % scale)
            mesh.init()

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
            p_in = Expression("sin(3.0*t)", t=0.0)

            # Define boundary conditions
            noslip = DirichletBC(V, (0, 0),
                                 "on_boundary && \
                                  (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                                  (x[0] > 0.5 - DOLFIN_EPS && x[1] > 0.5 - DOLFIN_EPS))")
            inflow = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
            outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
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

        if preassemble and symmetric:
            with self.timed_region('matrix assembly'):
                # Assemble matrices
                A1 = PETScMatrix()
                A2 = PETScMatrix()
                A3 = PETScMatrix()
                assembler1 = SystemAssembler(a1, L1, bcu)
                assembler2 = SystemAssembler(a2, L2, bcp)
                assembler3 = SystemAssembler(a3, L3, bcu)
                assembler1.assemble(A1)
                assembler2.assemble(A2)
                assembler3.assemble(A3)
                b1 = PETScVector()
                b2 = PETScVector()
                b3 = PETScVector()
        if preassemble and not symmetric:
            with self.timed_region('matrix assembly'):
                # Assemble matrices
                A1 = assemble(a1)
                A2 = assemble(a2)
                A3 = assemble(a3)

        if save:
            # Create files for storing solution
            ufile = File("vtk/dolfin_velocity.pvd")
            pfile = File("vtk/dolfin_pressure.pvd")

        vparams = {'linear_solver': 'gmres',
                   'preconditioner': 'bjacobi'}
        pparams = {'linear_solver': 'gmres',
                   'preconditioner': 'bjacobi'}

        with self.timed_region('timestepping'):
            # Time-stepping
            t = dt
            while t < T + DOLFIN_EPS:
                # Update pressure boundary condition
                p_in.t = t

                # Compute tentative velocity step
                begin("Computing tentative velocity")
                if preassemble:
                    with self.timed_region('tentative velocity RHS'):
                        if symmetric:
                            assembler1.assemble(b1)
                        else:
                            b1 = assemble(L1)
                            [bc.apply(A1, b1) for bc in bcu]
                    with self.timed_region('tentative velocity solve'):
                        solve(A1, u1.vector(), b1, "gmres", "bjacobi")
                else:
                    with self.timed_region('tentative velocity solve'):
                        solve(a1 == L1, u1, bcs=bcu, solver_parameters=vparams)
                end()

                # Pressure correction
                begin("Computing pressure correction")
                if preassemble:
                    with self.timed_region('pressure correction RHS'):
                        if symmetric:
                            assembler2.assemble(b2)
                            psolver = "cg"
                        else:
                            b2 = assemble(L2)
                            [bc.apply(A2, b2) for bc in bcp]
                            psolver = "gmres"
                    with self.timed_region('pressure correction solve'):
                        solve(A2, p1.vector(), b2, psolver, "bjacobi")
                else:
                    with self.timed_region('pressure correction solve'):
                        solve(a2 == L2, p1, bcs=bcp, solver_parameters=pparams)
                end()

                # Velocity correction
                begin("Computing velocity correction")
                if preassemble:
                    with self.timed_region('velocity correction RHS'):
                        if symmetric:
                            assembler3.assemble(b3)
                        else:
                            b3 = assemble(L3)
                            [bc.apply(A3, b3) for bc in bcu]
                    with self.timed_region('velocity correction solve'):
                        solve(A3, u1.vector(), b3, "gmres", "bjacobi")
                else:
                    with self.timed_region('velocity correction solve'):
                        solve(a3 == L3, u1, bcs=bcu, solver_parameters=vparams)
                end()

                if save:
                    # Save to file
                    ufile << u1
                    pfile << p1

                if compute_norms:
                    nu1, np1 = norm(u1), norm(p1)
                    if MPI.rank(mpi_comm_world()) == 0:
                        print t, 'u1:', nu1, 'p1:', np1

                # Move to next time step
                u0.assign(u1)
                t += dt
        t = timings(True)
        for task in ['Assemble system', 'Build sparsity', 'PETSc Krylov solver']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    DolfinNavierStokes().main()

    # Output VTU files
    # DolfinNavierStokes().navier_stokes(save=True)
