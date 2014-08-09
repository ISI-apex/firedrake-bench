from pybench import Benchmark

scale = [1.0, 0.71, 0.5, 0.35, 0.25, 0.18, 0.125]
cells = dict(zip(scale, [42254, 82072, 169418, 337266, 679624, 1309528, 2716428]))
dofs = dict(zip(scale, [21119, 41028, 84701, 168625, 339804, 656622, 1358206]))
regions = ['setup', 'timestepping', 'p', 'phi']


class Wave(Benchmark):
    method = 'wave'
    benchmark = 'Wave'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

# Weak scaling sizes
cells[0.2] = 1059980
cells[0.16] = 1617232
cells[0.14] = 2172644
cells[0.1] = 4247104
dofs[0.2] = 529982
dofs[0.16] = 808608
dofs[0.14] = 1086314
dofs[0.1] = 2123544

if __name__ == '__main__':
    from sys import argv
    b = Wave()
    b.combine_series([('np', [1]), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=cells, kinds='plot,loglog', groups=['variant'],
           title='Explicit wave equation (single core, 2D, mass lumping: %(lump_mass)s)')
    if len(argv) > 1 and argv[1] == 'weak':
        b = Wave(benchmark='WaveWeak')
        b.combine_series([('np', map(int, argv[2:])), ('variant', ['Firedrake'])], filename='Wave')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (weak scaling, single node, 2D)')
    elif len(argv) > 1:
        b = Wave(benchmark='WaveParallel')
        b.combine_series([('np', map(int, argv[1:])), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='Wave')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (single node, 2D, mesh scaling %(scale)s)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Explicit wave equation (single node, 2D, mesh scaling %(scale)s)')
