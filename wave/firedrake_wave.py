from wave import Wave
from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
from pyop2 import __version__ as pyop2_version

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True

Wave.meta.update({'coffee': parameters["coffee"],
                  'firedrake': firedrake_version,
                  'pyop2': pyop2_version})


class FiredrakeWave(Wave):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}

    @memoize
    def make_mesh(self, scale):
        return Mesh("meshes/wave_tank_%s.msh" % scale)

    def wave(self, scale=1.0, lump_mass=True, N=100):
        self.series['scale'] = scale
        with self.timed_region('mesh'):
            mesh = self.make_mesh(scale)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, 'Lagrange', 1)
            p = Function(V)
            phi = Function(V, name="phi")

            u = TrialFunction(V)
            v = TestFunction(V)

            bcval = Constant(0.0)
            bc = DirichletBC(V, bcval, 1)

            if lump_mass:
                Ml = assemble(1.0 / assemble(v*dx))

            dt = 0.001 * scale
            t = 0.0

            rhs = inner(grad(v), grad(phi)) * dx

        with self.timed_region('timestepping'):
            while t < N*dt:
                bcval.assign(sin(2*pi*5*t))

                with self.timed_region('phi'):
                    phi -= 0.5 * dt * p
                    phi.dat._force_evaluation()

                with self.timed_region('p'):
                    if lump_mass:
                        p += dt * Ml * assemble(rhs)
                        bc.apply(p)
                        p.dat._force_evaluation()
                    else:
                        solve(u * v * dx == v * p * dx + dt * rhs,
                              p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                                            'pc_type': 'sor',
                                                            'pc_sor_symmetric': True})

                with self.timed_region('phi'):
                    phi -= 0.5 * dt * p
                    phi.dat._force_evaluation()

                t += dt

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeWave().main()
