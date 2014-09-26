from navier_stokes import NavierStokes, cells, vertices
from pybench import timed
from firedrake import *
from firedrake.utils import memoize
from pyop2.profiling import get_timers

from firedrake_common import FiredrakeBenchmark

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True


class FiredrakeNavierStokes(FiredrakeBenchmark, NavierStokes):

    @memoize
    @timed
    def make_mesh(self, scale):
        return Mesh("meshes/lshape_%s.msh" % scale)

    def navier_stokes(self, scale=1.0, T=0.1, preassemble=True, save=False,
                      weak=False, compute_norms=False):
        if weak:
            self.series['weak'] = scale
            scale = round(scale/sqrt(op2.MPI.comm.size), 3)
            self.meta['scale'] = scale
        else:
            self.series['scale'] = scale
        self.meta['cells'] = cells[scale]
        self.meta['vertices'] = vertices[scale]
        t_, mesh = self.make_mesh(scale)
        self.register_timing('mesh', t_)

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
                A1 = assemble(a1, bcs=bcu)
                A2 = assemble(a2, bcs=bcp)
                A3 = assemble(a3, bcs=bcu)
                A1.M
                A2.M
                A3.M

        if save:
            # Create files for storing solution
            ufile = File("vtk/firedrake_velocity.pvd")
            pfile = File("vtk/firedrake_pressure.pvd")

        vparams = {'ksp_type': 'gmres',
                   'pc_type': 'bjacobi',
                   'sub_pc_type': 'ilu',
                   'ksp_rtol': 1e-6,
                   'ksp_atol': 1e-15}
        pparams = {'ksp_type': 'cg',
                   'pc_type': 'bjacobi',
                   'sub_pc_type': 'ilu',
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
                        b1.dat.data_ro
                    with self.timed_region('tentative velocity solve'):
                        solve(A1, u1, b1, solver_parameters=vparams)
                        u1.dat.data_ro
                else:
                    with self.timed_region('tentative velocity solve'):
                        solve(a1 == L1, u1, bcs=bcu, solver_parameters=vparams)
                        u1.dat.data_ro

                # Pressure correction
                info("Computing pressure correction")
                if preassemble:
                    with self.timed_region('pressure correction RHS'):
                        b2 = assemble(L2)
                        [bc.apply(A2, b2) for bc in bcp]
                        b2.dat.data_ro
                    with self.timed_region('pressure correction solve'):
                        solve(A2, p1, b2, solver_parameters=pparams)
                        p1.dat.data_ro
                else:
                    with self.timed_region('pressure correction solve'):
                        solve(a2 == L2, p1, bcs=bcp, solver_parameters=pparams)
                        p1.dat.data_ro

                # Velocity correction
                info("Computing velocity correction")
                if preassemble:
                    with self.timed_region('velocity correction RHS'):
                        b3 = assemble(L3)
                        [bc.apply(A3, b3) for bc in bcu]
                        b3.dat.data_ro
                    with self.timed_region('velocity correction solve'):
                        solve(A3, u1, b3, solver_parameters=vparams)
                        u1.dat.data_ro
                else:
                    with self.timed_region('velocity correction solve'):
                        solve(a3 == L3, u1, bcs=bcu, solver_parameters=vparams)
                        u1.dat.data_ro

                if save:
                    # Save to file
                    ufile << u1
                    pfile << p1

                if compute_norms:
                    nu1, np1 = norm(u1), norm(p1)
                    if op2.MPI.comm.rank == 0:
                        print t, 'u1:', nu1, 'p1:', np1

                # Move to next time step
                u0.assign(u1)
                t += dt
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    FiredrakeNavierStokes().main()

    # Output VTU files
    # FiredrakeNavierStokes().navier_stokes(save=True)
