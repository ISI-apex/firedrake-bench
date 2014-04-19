from time import clock
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}


def poisson(size, degree=1, dim=2):
    t_ = clock()
    mesh = make_mesh[dim](size)
    mesh.init()
    print dim, size, degree, 'mesh:', clock() - t_
    t_ = clock()
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
    g = Expression("sin(5*x[0])")
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Compute solution
    u = Function(V)
    print dim, size, degree, 'setup:', clock() - t_
    t_ = clock()
    A = assemble(a)
    bc.apply(A)
    print dim, size, degree, 'matrix:', clock() - t_
    t_ = clock()
    b = assemble(L)
    bc.apply(b)
    print dim, size, degree, 'rhs:', clock() - t_
    t_ = clock()
    solve(A, u.vector(), b, "cg", "default")
    print dim, size, degree, 'solve:', clock() - t_

if __name__ == '__main__':
    set_log_active(False)
    for degree in range(1, 4):
        for size in range(4, 7):
            poisson(2**size, degree=degree, dim=3)
