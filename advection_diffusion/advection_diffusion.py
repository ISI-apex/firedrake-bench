from pybench import Benchmark
from itertools import product

r0 = ['DOLFIN', 'Firedrake']
r1 = ['advection', 'diffusion']
r2 = ['RHS', 'solve']


class AdvectionDiffusion(Benchmark):

    params = [('degree', range(1, 4)),
              ('scale', [1.0, 0.71, 0.5, 0.35, 0.25])]
    meta = {'cells': [26386, 52166, 105418, 216162, 422660],
            'dofs': [13192, 26082, 52708, 108080, 211327]}
    method = 'advection_diffusion'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = map(' '.join, product(r1, r2))

if __name__ == '__main__':
    import sys
    regions = map(' '.join, product(r0, r1, r2))
    b = AdvectionDiffusion()
    b.combine({'FiredrakeAdvectionDiffusion_np1': 'Firedrake',
               'DolfinAdvectionDiffusion_np1': 'DOLFIN'})
    b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog')
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
           kinds='bar,barlog')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = AdvectionDiffusion(name='DolfinAdvectionDiffusionParallel')
        b.combine_series([('np', np)], filename='DolfinAdvectionDiffusion')
        b.save()
        b = AdvectionDiffusion(name='FiredrakeAdvectionDiffusionParallel')
        b.combine_series([('np', np)], filename='FiredrakeAdvectionDiffusion')
        b.save()
        b = AdvectionDiffusion(name='AdvectionDiffusionParallel')
        b.combine({'FiredrakeAdvectionDiffusionParallel': 'Firedrake',
                   'DolfinAdvectionDiffusionParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
