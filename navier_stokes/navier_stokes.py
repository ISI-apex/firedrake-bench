from pybench import Benchmark
from itertools import product

r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
r2 = ['RHS', 'solve']


class NavierStokes(Benchmark):

    params = [('scale', [0.8, 0.56, 0.4, 0.28, 0.2])]
    meta = {'cells': [30906, 63432, 124390, 253050, 496156],
            'dofs': [15451, 31714, 62193, 126523, 248076]}
    method = 'navier_stokes'
    name = 'NavierStokes'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'matrix assembly': {'marker': 'p'},
                 'timestepping': {'marker': 'o'},
                 'tentative velocity RHS': {'marker': '^'},
                 'tentative velocity solve': {'marker': 'v'},
                 'pressure correction RHS': {'marker': 's'},
                 'pressure correction solve': {'marker': 'D'},
                 'velocity correction RHS': {'marker': '>'},
                 'velocity correction solve': {'marker': '<'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = ['matrix assembly'] + map(' '.join, product(r1, r2))

if __name__ == '__main__':
    import sys
    regions = map(' '.join, product(r1, r2))
    b = NavierStokes()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'],
           title='Navier-Stokes (single core, 2D, P2-P1 discretisation)')
    for r in r2:
        b.plot(xaxis='scale', regions=map(' '.join, product(r1, [r])),
               xlabel='mesh size (cells)', figname='NavierStokes_' + r,
               xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'],
               title='Navier-Stokes %s (single core, 2D, P2-P1 discretisation)' % r)
    if len(sys.argv) > 1:
        tsuff = ' (single node, 2D, P2-P1 discretisation, mesh scaling: %(scale)s)'
        np = map(int, sys.argv[1:])
        b = NavierStokes(name='NavierStokesParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='NavierStokes')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Navier-Stokes' + tsuff)
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Navier-Stokes' + tsuff)
        for r in r2:
            b.plot(xaxis='np', regions=map(' '.join, product(r1, [r])),
                   xlabel='Number of processors', figname='NavierStokesParallel_' + r,
                   kinds='plot,loglog', groups=['variant'],
                   title='Navier-Stokes ' + r + tsuff)
            b.plot(xaxis='np', regions=map(' '.join, product(r1, [r])),
                   xlabel='Number of processors', figname='NavierStokesParallel_' + r,
                   kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
                   ylabel='Speedup relative to DOLFIN on 1 core',
                   title='Navier-Stokes ' + r + tsuff)
