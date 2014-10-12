from cahn_hilliard import CahnHilliard, lmbda, dt, theta
from firedrake import *
from pyop2.profiling import get_timers, timing

from firedrake_common import FiredrakeBenchmark

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True


class FiredrakeCahnHilliard(FiredrakeBenchmark, CahnHilliard):

    def cahn_hilliard(self, size=96, steps=10, degree=1, pc='fieldsplit',
                      inner_ksp='preonly', ksp='gmres', maxit=1, weak=False,
                      measure_overhead=False, save=False, compute_norms=True,
                      verbose=False):
        if weak:
            self.series['weak'] = size
            size = int((size*op2.MPI.comm.size)**0.5)
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.meta['cells'] = 2*size**2
        self.meta['vertices'] = (size+1)**2
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
        if verbose:
            params['ksp_monitor'] = True
            params['snes_view'] = True
            params['snes_monitor'] = True

        t_, mesh = self.make_mesh(size)
        self.register_timing('mesh', t_)

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

            with self.timed_region('initial condition'):
                # Create intial conditions and interpolate
                init_code = "A[0] = 0.63 + 0.02*(0.5 - (double)random()/RAND_MAX);"
                user_code = """int __rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &__rank);
                srandom(2 + __rank);"""
                par_loop(init_code, direct, {'A': (u[0], WRITE)},
                         headers=["#include <stdlib.h>"], user_code=user_code)
                u.dat.data_ro

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
                pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, pc_schur)

            # Output file
            if save:
                file = File("vtk/firedrake_cahn_hilliard_%d.pvd" % size)

        if measure_overhead:
            for _ in range(100):
                u0.assign(u)
                solver.solve()
            print "Assembly overhead:", timing("Assemble cells", total=False)
            print "Solver overhead:", timing("SNES solver execution", total=False)
            return
        with self.timed_region('timestepping'):
            for step in range(steps):
                with self.timed_region('timestep_%s' % step):
                    u0.assign(u)
                    solver.solve()
                if save:
                    file << (u.split()[0], step)
                if compute_norms:
                    nu = norm(u)
                    if op2.MPI.comm.rank == 0:
                        print step, 'L2(u):', nu
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
