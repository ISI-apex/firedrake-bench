from cahn_hilliard import CahnHilliard, lmbda, dt, theta
from firedrake import *
from pyop2.profiling import get_timers


class FiredrakeCahnHilliard(CahnHilliard):
    series = {'np': op2.MPI.comm.size}
    plotstyle = {'total': {'marker': '*', 'linestyle': '-'},
                 'mesh': {'marker': 's', 'linestyle': '-'},
                 'setup': {'marker': 'D', 'linestyle': '-'},
                 'timestepping': {'marker': 'o', 'linestyle': '-'}}

    def cahn_hilliard(self, size=96, steps=10, degree=1, pc='fieldsplit',
                      inner_ksp='preonly', ksp='gmres', maxit=1,
                      save=False, compute_norms=False):
        params = {'pc_type': pc,
                  'ksp_type': ksp,
                  'snes_rtol': 1e-9,
                  'snes_atol': 1e-10,
                  'snes_stol': 1e-16,
                  'snes_linesearch_type': 'basic',
                  'snes_linesearch_max_it': 1,
                  'ksp_rtol': 1e-6,
                  'ksp_atol': 1e-15,
                  'pc_fieldsplit_type': 'schur',
                  'pc_fieldsplit_schur_factorization_type': 'lower',
                  'pc_fieldsplit_schur_precondition': 'user',
                  'fieldsplit_0_ksp_type': inner_ksp,
                  'fieldsplit_0_ksp_max_it': maxit,
                  'fieldsplit_0_pc_type': 'hypre',
                  'fieldsplit_1_ksp_type': inner_ksp,
                  'fieldsplit_1_ksp_max_it': maxit,
                  'fieldsplit_1_pc_type': 'mat'}

        with self.timed_region('mesh'):
            # Create mesh and define function spaces
            mesh = UnitSquareMesh(size, size)

        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "Lagrange", degree)
            ME = V*V

            # Define trial and test functions
            du = TrialFunction(ME)
            q, v = TestFunctions(ME)

            # Define functions
            u = Function(ME)   # current solution
            u0 = Function(ME)  # solution from previous converged step

            # Split mixed functions
            dc, dmu = split(du)
            c, mu = split(u)
            c0, mu0 = split(u0)

            # Create intial conditions and interpolate
            init_code = """void u_init(double A[1]) {
              A[0] = 0.63 + 0.02*(0.5 - (double)random()/RAND_MAX);
            }"""
            user_code = """int __rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &__rank);
            srandom(2 + __rank);"""
            u_init = op2.Kernel(init_code, "u_init",
                                headers=["#include <stdlib.h>"],
                                user_code=user_code)
            op2.par_loop(u_init, u.function_space().node_set[0],
                         u.dat[0](op2.WRITE))
            u.dat._force_evaluation()

            # Compute the chemical potential df/dc
            c = variable(c)
            f = 100*c**2*(1-c)**2
            dfdc = diff(f, c)

            mu_mid = (1.0-theta)*mu0 + theta*mu

            # Weak statement of the equations
            F0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
            F1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
            F = F0 + F1

            # Compute directional derivative about u in the direction of du (Jacobian)
            J = derivative(F, u, du)

            problem = NonlinearVariationalProblem(F, u, J=J)
            solver = NonlinearVariationalSolver(problem, solver_parameters=params)

            if pc in ['fieldsplit', 'ilu']:
                sigma = 100
                # PC for the Schur complement solve
                trial = TrialFunction(V)
                test = TestFunction(V)
                mass = assemble(inner(trial, test)*dx).M.handle
                a = 1
                c = (dt * lmbda)/(1+dt * sigma)
                hats = assemble(sqrt(a) * inner(trial, test)*dx + sqrt(c)*inner(grad(trial), grad(test))*dx).M.handle

                from firedrake.petsc import PETSc
                ksp_hats = PETSc.KSP()
                ksp_hats.create()
                ksp_hats.setOperators(hats)
                opts = PETSc.Options()

                opts['ksp_type'] = inner_ksp
                opts['ksp_max_it'] = maxit
                opts['pc_type'] = 'hypre'
                ksp_hats.setFromOptions()

                class SchurInv(object):
                    def mult(self, mat, x, y):
                        tmp1 = y.duplicate()
                        tmp2 = y.duplicate()
                        ksp_hats.solve(x, tmp1)
                        mass.mult(tmp1, tmp2)
                        ksp_hats.solve(tmp2, y)

                pc_schur = PETSc.Mat()
                pc_schur.createPython(mass.getSizes(), SchurInv())
                pc_schur.setUp()
                pc = solver.snes.ksp.pc
                pc.setFieldSplitSchurPrecondition(PETSc.PC.SchurPreType.USER, pc_schur)

            # Output file
            if save:
                file = File("vtk/firedrake_cahn_hilliard_%d.pvd" % size)

        with self.timed_region('timestepping'):
            # Step in time
            t = 0.0
            T = steps*dt
            while (t < T):
                t += dt
                u0.assign(u)
                solver.solve()
                if save:
                    file << (u.split()[0], t)
                if compute_norms:
                    nu = norm(u)
                    if op2.MPI.comm.rank == 0:
                        print t, 'L2(u):', nu
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    FiredrakeCahnHilliard().main()

    # Output VTU files
    # FiredrakeCahnHilliard().cahn_hilliard(save=True)
