from wave import Wave
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

PETScOptions.set("pc_sor_symmetric", True)


class DolfinWave(Wave):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}

    def wave(self, scale=1.0, lump_mass=True, N=100, save=False):
        params = {'linear_solver': 'cg',
                  'preconditioner': 'sor'}
        self.series['scale'] = scale
        with self.timed_region('mesh'):
            mesh = Mesh("meshes/wave_tank_%s.xml.gz" % scale)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, 'Lagrange', 1)
            p = Function(V)
            phi = Function(V, name="phi")

            u = TrialFunction(V)
            v = TestFunction(V)

            # Define Dirichlet boundary (x = 0)
            def boundary(x):
                return x[0] < DOLFIN_EPS

            bcval = Constant(0.0)
            bc = DirichletBC(V, bcval, boundary)

            if lump_mass:
                Ml = 1.0 / assemble(v*dx).array()

            dt = 0.001 * scale
            t = 0.0

            rhs = inner(grad(v), grad(phi)) * dx

            if save:
                outfile = File("vtk/dolfin_wave_%s.pvd" % scale)
                outfile << phi

        with self.timed_region('timestepping'):
            while t < N*dt:
                bcval.assign(sin(2*pi*5*t))

                with self.timed_region('phi'):
                    phi.vector().axpy(-0.5 * dt, p.vector())

                with self.timed_region('p'):
                    if lump_mass:
                        p.vector().add_local(dt * Ml * assemble(rhs).array())
                        bc.apply(p.vector())
                    else:
                        solve(u * v * dx == v * p * dx + dt * rhs,
                              p, bcs=bc, solver_parameters=params)

                with self.timed_region('phi'):
                    phi.vector().axpy(-0.5 * dt, p.vector())

                t += dt
                if save:
                    outfile << phi

if __name__ == '__main__':
    set_log_active(False)
    from ffc.log import set_level
    set_level('ERROR')

    DolfinWave().main()
