import sys
import time

from mpi4py import MPI
from firedrake import *

from firedrake_cahn_hilliard_problem import CahnHilliardProblem

size = int(sys.argv[1])
fout_name = sys.argv[2]

steps=1
degree=1
pc='fieldsplit'
inner_ksp='preonly'
ksp='gmres'
maxit=1
save=False
compute_norms=True
verbose=False

comm = MPI.COMM_WORLD

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

params = CahnHilliardProblem.get_solve_params(verbose=verbose, \
        pc=pc, ksp=ksp, inner_ksp=inner_ksp, maxit=maxit)

time_mesh_begin = time.time()
mesh = CahnHilliardProblem.make_mesh(size)
time_mesh_end = time.time()

time_setup_begin = time.time()
u, u0, solver = CahnHilliardProblem.do_setup(mesh, pc, degree=degree,
        dt=dt, theta=theta, lmbda=lmbda, params=params)
time_setup_end = time.time()

# Output file
if save:
    file = File("vtk/firedrake_cahn_hilliard_%d.pvd" % size)
else:
    file = None

time_solve_begin = time.time()
CahnHilliardProblem.do_solve(u, u0, solver, steps,
        compute_norms=compute_norms, out_file=file)
time_solve_end = time.time()

if comm.rank == 0:
    from collections import OrderedDict
    times = OrderedDict()
    times["mesh"] = size
    times["mesh_s"]  = time_mesh_end - time_mesh_begin
    times["setup_s"] = time_setup_end - time_setup_begin
    times["solve_s"] = time_solve_end - time_solve_begin

    # note: if you open this earlier, the FD breaks somehow (???)
    fout = open(fout_name, "w")
    print(",".join(times.keys()), file=fout)
    print(",".join([str(v) for v in times.values()]), file=fout)
    fout.close()

    for step, t in times.items():
        print("%20s: %8.2f" % (step, t))
