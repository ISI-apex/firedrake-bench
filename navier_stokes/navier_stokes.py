from pybench import Benchmark
from itertools import product

r0 = ['DOLFIN', 'Firedrake']
r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
r2 = ['RHS', 'solve']


class NavierStokes(Benchmark):

    params = [('scale', [0.8, 0.56, 0.4, 0.28, 0.2])]
    meta = {'cells': [30906, 63432, 124390, 253050, 496156],
            'dofs': [15451, 31714, 62193, 126523, 248076]}
    method = 'navier_stokes'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = ['matrix assembly'] + map(' '.join, product(r1, r2))

if __name__ == '__main__':
    import sys
    regions = map(' '.join, product(r0, r1, r2))
    b = NavierStokes()
    b.combine({'FiredrakeNavierStokes_np1': 'Firedrake',
               'DolfinNavierStokes_np1': 'DOLFIN'})
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = NavierStokes(name='DolfinNavierStokesParallel')
        b.combine_series([('np', np)], filename='DolfinNavierStokes')
        b.save()
        b = NavierStokes(name='FiredrakeNavierStokesParallel')
        b.combine_series([('np', np)], filename='FiredrakeNavierStokes')
        b.save()
        b = NavierStokes(name='NavierStokesParallel')
        b.combine({'FiredrakeNavierStokesParallel': 'Firedrake',
                   'DolfinNavierStokesParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', wscale=0.7)
