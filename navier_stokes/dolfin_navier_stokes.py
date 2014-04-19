from time import clock
from dolfin import *

# Don't reorder the mesh
parameters["reorder_dofs_serial"] = False


def navier_stokes(x, T=3, preassemble=True, save=False):
    # Load mesh from file
    t_ = clock()
    mesh = Mesh("lshape_%s.xml.gz" % scale)
    mesh.init()
    print scale, 'mesh:', clock() - t_
    t_ = clock()

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
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
    print scale, 'setup:', clock() - t_
    t_ = clock()

    if preassemble:
        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

    print scale, 'matrix assembly:', clock() - t_
    t_ = clock()

    if save:
        # Create files for storing solution
        ufile = File("results/dolfin_velocity.pvd")
        pfile = File("results/dolfin_pressure.pvd")

    # Time-stepping
    t = dt
    while t < T + DOLFIN_EPS:
        print "t =", t
        print 'u0', norm(u0)

        # Update pressure boundary condition
        p_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        if preassemble:
            b1 = assemble(L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, u1.vector(), b1, "gmres", "ilu")
        else:
            solve(a1 == L1, u1, bcs=bcu,
                  solver_parameters={"linear_solver": "gmres",
                                     "preconditioner": "ilu"})
        print 'u1', norm(u1)
        end()

        # Pressure correction
        begin("Computing pressure correction")
        if preassemble:
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcp]
            solve(A2, p1.vector(), b2, "cg", "default")
        else:
            solve(a2 == L2, p1, bcs=bcp,
                  solver_parameters={"linear_solver": "cg",
                                     "preconditioner": "default"})
        print 'p1', norm(p1)
        end()

        # Velocity correction
        begin("Computing velocity correction")
        if preassemble:
            b3 = assemble(L3)
            [bc.apply(A3, b3) for bc in bcu]
            solve(A3, u1.vector(), b3, "gmres", "ilu")
        else:
            solve(a3 == L3, u1, bcs=bcu,
                  solver_parameters={"linear_solver": "gmres",
                                     "preconditioner": "ilu"})
        print 'u1', norm(u1)
        end()

        if save:
            # Save to file
            ufile << u1
            pfile << p1

        # Move to next time step
        u0.assign(u1)
        t += dt

    print scale, 'time stepping', clock() - t_

if __name__ == '__main__':
    set_log_active(False)
    for scale in (2, 1, 0.5):
        navier_stokes(scale, 1)
