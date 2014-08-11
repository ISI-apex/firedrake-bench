from advection_diffusion import AdvectionDiffusion
from pybench import timed
from dolfin import *

make_mesh = {2: lambda x: UnitSquareMesh(x, x),
             3: lambda x: UnitCubeMesh(x, x, x)}

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Tune AMG parameters
PETScOptions.set('pc_hypre_boomeramg_strong_threshold', 0.75)
PETScOptions.set('pc_hypre_boomeramg_agg_nl', 2)


@timed
def make_mesh(dim, x):
    mesh = UnitSquareMesh(x, x) if dim == 2 else UnitCubeMesh(x, x, x)
    mesh.init()
    return mesh


class DolfinAdvectionDiffusion(AdvectionDiffusion):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}
    meta = {'dolfin_version': dolfin_version(),
            'dolfin_commit': git_commit_hash()}
    meshes = {}

    def advection_diffusion(self, size=64, degree=1, dim=2,
                            dt=0.0001, T=0.01, Tend=0.011, diffusivity=0.1,
                            advection=True, diffusion=True, weak=False,
                            print_norm=False, preassemble=True, pc='amg'):
        if weak:
            size = int((1e4*MPI.size(mpi_comm_world()))**(1./dim))
            self.meta['size'] = size
        else:
            self.series['size'] = size
        self.meta['cells'] = (2 if dim == 2 else 6)*size**dim
        self.meta['dofs'] = (size+1)**dim
        solver_parameters = {'linear_solver': 'cg', 'preconditioner': pc}
        if (dim, size) in self.meshes:
            t_, mesh = self.meshes[dim, size]
        else:
            t_, mesh = make_mesh(dim, size)
            self.meshes[dim, size] = t_, mesh
        self.register_timing('mesh', t_)

        with self.timed_region('setup'):
            V = FunctionSpace(mesh, "CG", degree)
            U = VectorFunctionSpace(mesh, "CG", degree)

            p = TrialFunction(V)
            q = TestFunction(V)
            t = Function(V)
            u = Function(U)

            adv = p * q * dx
            adv_rhs = (q * t + dt * dot(grad(q), u) * t) * dx

            d = -dt * diffusivity * dot(grad(q), grad(p)) * dx

            diff = adv - 0.5 * d
            diff_rhs = action(adv + 0.5 * d, t)

            # Set initial condition:
            # A*(e^(-r^2/(4*D*T)) / (4*pi*D*T))
            # with normalisation A = 0.1, diffusivity D = 0.1
            r2 = "(pow(x[0]-(0.45+%(T)f), 2.0) + pow(x[1]-0.5, 2.0))"
            fexpr = "0.1 * (exp(-" + r2 + "/(0.4*%(T)f)) / (0.4*pi*%(T)f))"
            t.interpolate(Expression(fexpr % {'T': T}))
            u.interpolate(Expression(('1.0', '0.0')))

        if preassemble:
            if advection:
                with self.timed_region('advection matrix'):
                    A = assemble(adv)
            if diffusion:
                with self.timed_region('diffusion matrix'):
                    D = assemble(diff)

        with self.timed_region('timestepping'):
            while T < Tend:

                # Advection
                if advection:
                    if preassemble:
                        with self.timed_region('advection RHS'):
                            b = assemble(adv_rhs)
                        with self.timed_region('advection solve'):
                            solve(A, t.vector(), b, "cg", pc)
                    else:
                        solve(adv == adv_rhs, t, solver_parameters=solver_parameters)

                # Diffusion
                if diffusion:
                    if preassemble:
                        with self.timed_region('diffusion RHS'):
                            b = assemble(diff_rhs)
                        with self.timed_region('diffusion solve'):
                            solve(D, t.vector(), b, "cg", pc)
                    else:
                        solve(diff == diff_rhs, t, solver_parameters=solver_parameters)

                T = T + dt

        # Analytical solution
        a = Function(V)
        a.interpolate(Expression(fexpr % {'T': T}))
        l2 = sqrt(assemble(dot(t - a, t - a) * dx))
        if print_norm and MPI.rank(mpi_comm_world()) == 0:
            print 'L2 error norm:', l2
        t = timings(True)
        for task in ['Assemble cells', 'Build sparsity', 'PETSc Krylov solver']:
            self.register_timing(task, float(t.get(task, 'Total time')))

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinAdvectionDiffusion().main()
