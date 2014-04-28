from pybench import Benchmark


class NavierStokes(Benchmark):

    params = [('scale', [0.8, 0.56, 0.4, 0.28, 0.2]),
              ('preassemble', [True, False])]
    meta = {'cells': [30906, 63432, 124390, 253050, 496156, 1013484, 1981076],
            'dofs': [15451, 31714, 62193, 126523, 248076, 506740, 990536]}
    method = 'navier_stokes'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    from itertools import product
    r0 = ['DOLFIN', 'Firedrake']
    r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
    r2 = ['RHS', 'solve']
    regions = map(' '.join, product(r0, r1, r2))
    b = NavierStokes(name='DolfinNavierStokesParallel')
    b.combine_series([('np', [1, 2, 3])], filename='DolfinNavierStokes')
    b.save()
    b = NavierStokes(name='FiredrakeNavierStokesParallel')
    b.combine_series([('np', [1, 2, 3])], filename='FiredrakeNavierStokes')
    b.save()
    b = NavierStokes()
    b.combine({'FiredrakeNavierStokes_np1': 'Firedrake', 'DolfinNavierStokes_np1': 'DOLFIN'})
    b.plot(xaxis='scale', regions=regions)
    b = NavierStokes(name='NavierStokesParallel')
    b.combine({'FiredrakeNavierStokesParallel': 'Firedrake', 'DolfinNavierStokesParallel': 'DOLFIN'})
    b.plot(xaxis='np', regions=regions)
