from pybench import Benchmark

scale = [1.0, 0.71, 0.5, 0.35, 0.25]
regions = ['setup', 'timestepping', 'p', 'phi']


class Wave(Benchmark):
    meta = {'cells': [42254, 82072, 169418, 337266, 679624],
            'dofs': [21119, 41028, 84701, 168625, 339804]}
    method = 'wave'
    benchmark = 'Wave'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    import sys
    b = Wave()
    b.combine_series([('np', [1]), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'],
           title='Explicit wave equation (single core, 2D)')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = Wave(benchmark='WaveParallel')
        b.combine_series([('np', np), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='Wave')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (single node, 2D, mesh scaling %(scale)s)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Explicit wave equation (single node, 2D, mesh scaling %(scale)s)')
