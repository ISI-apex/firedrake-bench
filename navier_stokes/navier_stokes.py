from pybench import Benchmark
from itertools import product

scale = [1.0, 0.707, 0.5, 0.354, 0.25, 0.177, 0.125]
cells = dict(zip(scale, [19810, 39736, 79034, 158960, 318030, 631774, 1271436]))
vertices = dict(zip(scale, [9906, 19869, 39518, 79481, 159016, 315888, 635719]))
r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
r2 = ['RHS', 'solve']

# Weak scaling sizes
cells[0.577] = 59614
cells[0.408] = 119812
cells[0.316] = 198694
cells[0.289] = 238048
cells[0.204] = 478240
cells[0.158] = 793468
cells[0.144] = 957686
cells[0.102] = 1905144
cells[0.072] = 3828300
cells[0.051] = 7621454
cells[0.036] = 15302786
cells[0.026] = 29377542
vertices[0.577] = 29805
vertices[0.408] = 59904
vertices[0.316] = 99345
vertices[0.289] = 119022
vertices[0.204] = 239118
vertices[0.158] = 396732
vertices[0.144] = 478841
vertices[0.102] = 952570
vertices[0.072] = 1914148
vertices[0.051] = 3810725
vertices[0.036] = 7651391
vertices[0.026] = 14688769


class NavierStokes(Benchmark):

    method = 'navier_stokes'
    benchmark = 'NavierStokes'
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
    b.combine_series([('np', [1]), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])])
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
        b = NavierStokes(benchmark='NavierStokesParallel')
        b.combine_series([('np', np), ('scale', scale), ('variant', ['Firedrake', 'DOLFIN'])],
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
