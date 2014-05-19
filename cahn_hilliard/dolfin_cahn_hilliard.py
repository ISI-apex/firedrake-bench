from cahn_hilliard import CahnHilliard, lmbda, dt, theta
import random
from dolfin import *


# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self):
        # DOLFIN 1.3
        # random.seed(2 + MPI.process_number())
        # DOLFIN master
        random.seed(2 + MPI.rank(mpi_comm_world()))

    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0

    def value_shape(self):
        return (2,)


# Class for interfacing with the Newton solver
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)

    def J(self, A, x):
        assemble(self.a, tensor=A)

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"


class DolfinCahnHilliard(CahnHilliard):
    series = {'np': MPI.size(mpi_comm_world())}

    def cahn_hilliard(self, size=96, steps=10, degree=1, save=False, pc='jacobi'):
        with self.timed_region('mesh'):
            # Create mesh and define function spaces
            mesh = UnitSquareMesh(size, size)
            mesh.init()

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
            u_init = InitialConditions()
            u.interpolate(u_init)
            u0.interpolate(u_init)

            # Compute the chemical potential df/dc
            c = variable(c)
            f = 100*c**2*(1-c)**2
            dfdc = diff(f, c)

            # mu_(n+theta)
            mu_mid = (1.0-theta)*mu0 + theta*mu

            # Weak statement of the equations
            L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
            L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
            L = L0 + L1

            # Compute directional derivative about u in the direction of du (Jacobian)
            a = derivative(L, u, du)

            # Create nonlinear problem and PETSc SNES solver
            problem = CahnHilliardEquation(a, L)
            solver = PETScSNESSolver()
            solver.parameters["linear_solver"] = "gmres"
            solver.parameters["preconditioner"] = pc
            solver.parameters["report"] = False
            solver.parameters["krylov_solver"]["report"] = False

            # Output file
            if save:
                file = File("vtk/dolfin_cahn_hilliard_%d.pvd" % size)

        with self.timed_region('timestepping'):
            # Step in time
            t = 0.0
            T = steps*dt
            while (t < T):
                t += dt
                u0.vector()[:] = u.vector()
                solver.solve(problem, u.vector())
                if save:
                    file << (u.split()[0], t)
        t = timings(True)
        for task in ['Assemble cells', 'Build sparsity', 'SNES solver execution']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    # Benchmark
    DolfinCahnHilliard().main()

    # Output VTU files
    # DolfinCahnHilliard().cahn_hilliard(save=True)
