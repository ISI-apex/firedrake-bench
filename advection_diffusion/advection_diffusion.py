from pybench import Benchmark

regions = ['advection RHS', 'diffusion RHS', 'advection solve', 'diffusion solve']


class AdvectionDiffusion(Benchmark):

    params = [('degree', range(1, 4)),
              ('scale', [1.0, 0.71, 0.5, 0.35, 0.25])]
    meta = {'cells': [26386, 52166, 105418, 216162, 422660],
            'dofs': [13192, 26082, 52708, 108080, 211327]}
    method = 'advection_diffusion'
    name = 'AdvectionDiffusion'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'advection matrix': {'marker': '>'},
                 'diffusion matrix': {'marker': '<'},
                 'timestepping': {'marker': 'o'},
                 'advection RHS': {'marker': '^'},
                 'diffusion RHS': {'marker': 'v'},
                 'advection solve': {'marker': 's'},
                 'diffusion solve': {'marker': 'D'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    import sys
    b = AdvectionDiffusion()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
           kinds='bar,barlog', groups=['variant'])
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = AdvectionDiffusion(name='AdvectionDiffusionParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='AdvectionDiffusion')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'])
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core')
