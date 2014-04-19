from time import clock
from firedrake import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


def poisson(size, degree=1, dim=2):
    t_ = clock()
    mesh = make_mesh[dim](size)
    print dim, size, degree, 'mesh:', clock() - t_
    t_ = clock()
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define boundary condition
    bc = DirichletBC(V, 0.0, [3, 4])

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)"))
    g = Function(V).interpolate(Expression("sin(5*x[0])"))
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    print dim, size, degree, 'setup:', clock() - t_
    t_ = clock()
    A = assemble(a, bcs=bc)
    print dim, size, degree, 'matrix:', clock() - t_
    t_ = clock()
    b = assemble(L)
    bc.apply(b)
    print dim, size, degree, 'rhs:', clock() - t_
    t_ = clock()
    solve(A, u, b, solver_parameters={"ksp_type": "cg"})
    u.dat.data
    print dim, size, degree, 'solve:', clock() - t_

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    for degree in range(1, 4):
        for size in range(4, 7):
            poisson(2**size, degree=degree, dim=3)
