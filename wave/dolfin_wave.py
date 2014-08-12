from wave import Wave, cells, dofs
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

PETScOptions.set("pc_sor_symmetric", True)


class DolfinWave(Wave):
    series = {'np': MPI.size(mpi_comm_world()), 'variant': 'DOLFIN'}
    meta = {'dolfin_version': dolfin_version(),
            'dolfin_commit': git_commit_hash()}

    def wave(self, scale=1.0, lump_mass=True, N=100, save=False, weak=False,
             verbose=False):
        params = {'linear_solver': 'cg',
                  'preconditioner': 'sor'}
        if weak:
            scale = round(0.5/sqrt(MPI.size(mpi_comm_world())), 2)
            self.meta['scale'] = scale
        else:
            self.series['scale'] = scale
        self.meta['cells'] = cells[scale]
        self.meta['dofs'] = dofs[scale]
        with self.timed_region('mesh'):
            mesh = Mesh("meshes/wave_tank_%s.xml.gz" % scale)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, 'Lagrange', 1)
            if verbose:
                print '[%d]' % MPI.rank(mpi_comm_world()), 'DOFs:', V.dofmap().global_dimension()
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
                        p.vector().apply("add")
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
