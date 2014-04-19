from time import clock
from firedrake import *


def adv_diff(size, advection=True, diffusion=True):
    dt = 0.0001
    T = 0.01

    t_ = clock()
    mesh = UnitSquareMesh(size, size)
    print size, 'mesh:', clock() - t_
    t_ = clock()

    V = FunctionSpace(mesh, "CG", 1)
    U = VectorFunctionSpace(mesh, "CG", 1)

    p = TrialFunction(V)
    q = TestFunction(V)
    t = Function(V)
    u = Function(U)

    diffusivity = 0.1

    adv = p * q * dx
    adv_rhs = (q * t + dt * dot(grad(q), u) * t) * dx

    d = -dt * diffusivity * dot(grad(q), grad(p)) * dx

    diff = adv - 0.5 * d
    diff_rhs = action(adv + 0.5 * d, t)

    if advection:
        A = assemble(adv)
    if diffusion:
        D = assemble(diff)

    # Set initial condition:
    # A*(e^(-r^2/(4*D*T)) / (4*pi*D*T))
    # with normalisation A = 0.1, diffusivity D = 0.1
    r2 = "(pow(x[0]-(0.45+%(T)f), 2.0) + pow(x[1]-0.5, 2.0))"
    fexpr = "0.1 * (exp(-" + r2 + "/(0.4*%(T)f)) / (0.4*pi*%(T)f))"
    t.interpolate(Expression(fexpr % {'T': T}))
    u.interpolate(Expression((1.0, 0.0)))

    print size, 'setup:', clock() - t_
    t_ = clock()

    while T < 0.02:

        # Advection
        if advection:
            b = assemble(adv_rhs)
            solve(A, t, b, solver_parameters={"ksp_type": "gmres", "pc_type": "ilu"})

        # Diffusion
        if diffusion:
            b = assemble(diff_rhs)
            solve(D, t, b, solver_parameters={"ksp_type": "gmres", "pc_type": "ilu"})

        T = T + dt

    # Analytical solution
    a = Function(V).interpolate(Expression(fexpr % {'T': T}))
    sqrt(assemble(dot(t - a, t - a) * dx))
    print size, 'time stepping:', clock() - t_

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    for size in range(5, 9):
        adv_diff(2**size)
