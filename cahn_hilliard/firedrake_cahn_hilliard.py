from time import clock
from firedrake import *

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson


def cahn_hilliard(x, steps, save=False):
    # Create mesh and define function spaces
    t_ = clock()
    mesh = UnitSquareMesh(x, x)
    print x, 'mesh:', clock() - t_
    t_ = clock()
    V = FunctionSpace(mesh, "Lagrange", 1)
    ME = V*V

    # Define trial and test functions
    du = TrialFunction(ME)
    q, v = TestFunctions(ME)

    # Define functions
    u = Function(ME)  # current solution
    u0 = Function(ME)  # solution from previous converged step

    # Split mixed functions
    dc, dmu = split(du)
    c, mu = split(u)
    c0, mu0 = split(u0)

    # Create intial conditions and interpolate
    u_init = op2.Kernel("""void u_init(double A[1]) {
  A[0] = 0.63 + 0.02*(0.5 - (double)random()/RAND_MAX);
}""", "u_init", headers=["#include <stdlib.h>"],
                        user_code="srandom(%d);" % (2 + op2.MPI.comm.rank))
    op2.par_loop(u_init, u.function_space().node_set[0], u.dat[0](op2.WRITE))

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
    solver = NonlinearVariationalSolver(problem,
                                        parameters={'ksp_type': 'gmres',
                                                    'pc_type': 'jacobi',
                                                    'pc_fieldsplit_type': 'schur',
                                                    'pc_fieldsplit_schur_fact_type': 'full',
                                                    'fieldsplit_0_ksp_type': 'cg',
                                                    'fieldsplit_1_ksp_type': 'cg',
                                                    'snes_rtol': 1e-9,
                                                    'snes_atol': 1e-10,
                                                    'snes_stol': 1e-16,
                                                    'ksp_rtol': 1e-6,
                                                    'ksp_atol': 1e-15,
                                                    'snes_linesearch_type': 'basic'})

    # Output file
    if save:
        file = File("firedrake_cahn_hilliard.pvd")

    print x, 'setup:', clock() - t_
    t_ = clock()
    # Step in time
    t = 0.0
    T = steps*dt
    while (t < T):
        t += dt
        u0.assign(u)
        solver.solve()
        if save:
            file << (u.split()[0], t)
    print x, 'time stepping for', steps, 'steps:', clock() - t_

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    cahn_hilliard(96, 50)
