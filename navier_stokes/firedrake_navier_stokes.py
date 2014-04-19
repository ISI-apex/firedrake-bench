from time import clock
from firedrake import *


def navier_stokes(scale, T=3, preassemble=True, save=False):
    # Load mesh from file
    t_ = clock()
    mesh = Mesh("lshape_%s.msh" % scale)
    print scale, 'mesh:', clock() - t_
    t_ = clock()

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
        ufile = File("results/firedrake_velocity.pvd")
        pfile = File("results/firedrake_pressure.pvd")

    vparams = {'ksp_type': 'gmres',
               'pc_type': 'ilu',
               'ksp_rtol': 1e-6,
               'ksp_atol': 1e-15}
    pparams = {'ksp_type': 'cg',
               'pc_type': 'ilu',
               'ksp_rtol': 1e-6,
               'ksp_atol': 1e-15}
    # Time-stepping
    t = dt
    while t < T + 1e-14:
        print "t =", t
        print 'u0', norm(u0)

        # Update pressure boundary condition
        p_in.assign(sin(3.0*t))

        # Compute tentative velocity step
        info("Computing tentative velocity")
        if preassemble:
            b1 = assemble(L1)
            [bc.apply(A1, b1) for bc in bcu]
            solve(A1, u1, b1, solver_parameters=vparams)
        else:
            solve(a1 == L1, u1, bcs=bcu, solver_parameters=vparams)
        print 'u1', norm(u1)

        # Pressure correction
        info("Computing pressure correction")
        if preassemble:
            b2 = assemble(L2)
            [bc.apply(A2, b2) for bc in bcp]
            solve(A2, p1, b2, solver_parameters=pparams)
        else:
            solve(a2 == L2, p1, bcs=bcp, solver_parameters=pparams)
        print 'p1', norm(p1)

        # Velocity correction
        info("Computing velocity correction")
        if preassemble:
            b3 = assemble(L3)
            [bc.apply(A3, b3) for bc in bcu]
            solve(A3, u1, b3, solver_parameters=vparams)
        else:
            solve(a3 == L3, u1, bcs=bcu, solver_parameters=vparams)
        print 'u1', norm(u1)

        if save:
            # Save to file
            ufile << u1
            pfile << p1

        # Move to next time step
        u0.assign(u1)
        t += dt

    print scale, 'time stepping', clock() - t_

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    for scale in (2, 1, 0.5):
        navier_stokes(scale, 1)
