import numpy as np
from wave import Wave, cells, vertices
from pybench import timed
from firedrake import *
from firedrake.utils import memoize
from pyop2.profiling import get_timers

from firedrake_common import FiredrakeBenchmark

parameters["coffee"]["licm"] = True
parameters["coffee"]["ap"] = True
parameters["assembly_cache"]["enabled"] = False


class FiredrakeWave(FiredrakeBenchmark, Wave):

    @memoize
    @timed
    def make_mesh(self, scale):
        return Mesh("meshes/wave_tank_%s.msh" % scale)

    def wave(self, scale=1.0, lump_mass=True, N=100, save=False, weak=False,
             verbose=False):
        if weak:
            scale = round(scale/sqrt(op2.MPI.comm.size), 3)
            self.meta['scale'] = scale
        else:
            self.series['scale'] = scale
        self.meta['cells'] = cells[scale]
        self.meta['vertices'] = vertices[scale]
        t_, mesh = self.make_mesh(scale)
        self.register_timing('mesh', t_)
        with self.timed_region('setup'):
            V = FunctionSpace(mesh, 'Lagrange', 1)
            total_dofs = np.zeros(1, dtype=int)
            op2.MPI.comm.Allreduce(np.array([V.dof_dset.size], dtype=int), total_dofs)
            self.meta['dofs'] = total_dofs[0]
            if verbose:
                print '[%d]' % op2.MPI.comm.rank, 'DOFs:', V.dof_dset.size
            p = Function(V)
            phi = Function(V, name="phi")

            u = TrialFunction(V)
            v = TestFunction(V)

            bcval = Constant(0.0)
            bc = DirichletBC(V, bcval, 1)

            if lump_mass:
                Ml = assemble(1.0 / assemble(v*dx))

            dt = 0.001 * scale
            dtc = Constant(dt)
            t = 0.0

            rhs = inner(grad(v), grad(phi)) * dx

            if save:
                outfile = File("vtk/firedrake_wave_%s.pvd" % scale)
                outfile << phi

        b = assemble(rhs)
        dphi = 0.5 * dtc * p
        dp = dtc * Ml * b
        with self.timed_region('timestepping'):
            while t < N*dt:
                bcval.assign(sin(2*pi*5*t))

                with self.timed_region('phi'):
                    phi -= dphi
                    phi.dat.data_ro

                with self.timed_region('p'):
                    if lump_mass:
                        assemble(rhs, tensor=b)
                        p += dp
                        bc.apply(p)
                        p.dat.data_ro
                    else:
                        solve(u * v * dx == v * p * dx + dtc * rhs,
                              p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                                            'pc_type': 'sor',
                                                            'pc_sor_symmetric': True})

                with self.timed_region('phi'):
                    phi -= dphi
                    phi.dat.data_ro

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
