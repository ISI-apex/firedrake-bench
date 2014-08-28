from cahn_hilliard import CahnHilliard, lmbda, dt, theta
import sys
import numpy as np
from petsc4py import PETSc
from pybench import timed
from dolfin import *


# Class representing the intial conditions
u_init_code = """
#include <stdlib.h>
namespace dolfin {

class InitialConditions : public Expression {
public:
  InitialConditions() : Expression(2) {
    srandom(2 + MPI::process_number());
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const {
    values[0] = 0.63 + 0.02*(0.5 - (double)random()/RAND_MAX);
    values[1] = 0.0;
  }
};

}
"""


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
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"


@timed
def make_mesh(x):
    mesh = UnitSquareMesh(x, x)
    mesh.init()
    return mesh


class DolfinCahnHilliard(CahnHilliard):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}
    meshes = {}

    def cahn_hilliard(self, size=96, steps=10, degree=1, pc='fieldsplit',
                      inner_ksp='preonly', ksp='gmres', maxit=1, weak=False,
                      save=False, compute_norms=True, verbose=False):
        if weak:
            size = int((size*MPI.size(mpi_comm_world()))**0.5)
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.meta['cells'] = 2*size**2
        self.meta['vertices'] = (size+1)**2
        if pc == 'fieldsplit':
            fs_petsc_args = [sys.argv[0]] + ("""
             --petsc.ok_ksp_type %(ksp)s
             --petsc.ok_snes_rtol 1e-9
             --petsc.ok_snes_atol 1e-10
             --petsc.ok_snes_stol 1e-16
             --petsc.ok_ksp_rtol 1e-6
             --petsc.ok_ksp_atol 1e-15

             --petsc.ok_pc_type fieldsplit
             --petsc.ok_pc_fieldsplit_type schur
             --petsc.ok_pc_fieldsplit_schur_factorization_type lower
             --petsc.ok_pc_fieldsplit_schur_precondition user

             --petsc.ok_fieldsplit_0_ksp_type %(inner_ksp)s
             --petsc.ok_fieldsplit_0_ksp_max_it %(maxit)d
             --petsc.ok_fieldsplit_0_pc_type hypre

             --petsc.ok_fieldsplit_1_ksp_type %(inner_ksp)s
             --petsc.ok_fieldsplit_1_ksp_max_it %(maxit)d
             --petsc.ok_fieldsplit_1_pc_type mat
             --petsc.ok_fieldsplit_1_hats_pc_type hypre
             --petsc.ok_fieldsplit_1_hats_ksp_type %(inner_ksp)s
             --petsc.ok_fieldsplit_1_hats_ksp_max_it %(maxit)d
             """ % {'ksp': ksp, 'inner_ksp': inner_ksp, 'maxit': maxit}).split()

            parameters.parse(fs_petsc_args)
        if verbose:
            PETScOptions.set('ok_ksp_monitor')
            PETScOptions.set('ok_snes_view')
            PETScOptions.set('ok_snes_monitor')
        PETScOptions.set("sub_pc_type", pc)
        if size in self.meshes:
            t_, mesh = self.meshes[size]
        else:
            t_, mesh = make_mesh(size)
            self.meshes[size] = t_, mesh
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
                u_init = Expression(cppcode=u_init_code)
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
            solver.parameters["preconditioner"] = "bjacobi"
            solver.parameters["report"] = False
            solver.parameters["krylov_solver"]["report"] = False
            solver.parameters["options_prefix"] = "ok"

            solver.init(problem, u.vector())
            snes = solver.snes()
            snes.setFromOptions()

            # Configure the FIELDSPLIT stuff.
            if pc == 'fieldsplit':
                sigma = 100
                pc = snes.ksp.pc

                fields = []
                for i in range(2):
                    subspace = SubSpace(ME, i)
                    subdofs = subspace.dofmap().dofs()
                    IS = PETSc.IS()
                    IS.createGeneral(subdofs.astype(np.int32))
                    name = str(i)
                    fields.append((name, IS))

                pc.setFieldSplitIS(*fields)

                trial = TrialFunction(V)
                test = TestFunction(V)
                mass = as_backend_type(assemble(inner(trial, test)*dx)).mat()

                a = 1
                c = (dt * lmbda)/(1 + dt * sigma)
                hats = as_backend_type(assemble(sqrt(a) * inner(trial, test)*dx + sqrt(c) * inner(grad(trial), grad(test))*dx)).mat()
                ksp_hats = PETSc.KSP()
                ksp_hats.create()
                ksp_hats.setOperators(hats)
                ksp_hats.setOptionsPrefix("ok_fieldsplit_1_hats_")
                ksp_hats.setFromOptions()
                ksp_hats.setUp()

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

                def monitor(snes, its, norm):
                    pc = snes.ksp.pc
                    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, pc_schur)

                snes.setMonitor(monitor)

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
                if compute_norms:
                    nu = norm(u)
                    if MPI.rank(mpi_comm_world()) == 0:
                        print t, 'L2(u):', nu
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
