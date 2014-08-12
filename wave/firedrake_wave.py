import numpy as np
from wave import Wave, cells, dofs
from pybench import timed
from firedrake import *
from firedrake import __version__ as firedrake_version
from firedrake.utils import memoize
from pyop2 import __version__ as pyop2_version
from pyop2.profiling import get_timers

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True


class FiredrakeWave(Wave):
    series = {'np': op2.MPI.comm.size, 'variant': 'Firedrake'}
    meta = {'coffee': parameters["coffee"],
            'firedrake': firedrake_version,
            'pyop2': pyop2_version}

    @memoize
    @timed
    def make_mesh(self, scale):
        return Mesh("meshes/wave_tank_%s.msh" % scale)

    def wave(self, scale=1.0, lump_mass=True, N=100, save=False, weak=False):
        if weak:
            scale = round(0.5/sqrt(op2.MPI.comm.size), 2)
            self.meta['scale'] = scale
        else:
            self.series['scale'] = scale
        self.meta['cells'] = cells[scale]
        self.meta['vertices'] = dofs[scale]
        t_, mesh = self.make_mesh(scale)
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, 'Lagrange', 1)
            total_dofs = np.zeros(1, dtype=int)
            op2.MPI.comm.Allreduce(np.array([V.dof_count], dtype=int), total_dofs)
            self.meta['dofs'] = total_dofs[0]
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

            if save:
                outfile = File("vtk/firedrake_wave_%s.pvd" % scale)
                outfile << phi

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
                if save:
                    outfile << phi
        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    FiredrakeWave().main()
