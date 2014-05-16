from pybench import Benchmark
from itertools import product

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(5)]
r0 = ['DOLFIN', 'Firedrake']
r1 = ['matrix assembly', 'rhs assembly', 'solve']


class Poisson(Benchmark):

    params = [('dim', [dim]),
              ('degree', range(1, 4)),
              ('size', sizes)]
    meta = {'cells': [6*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'poisson'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = r1

if __name__ == '__main__':
    import sys
    regions = map(' '.join, product(r0, r1))
    b = Poisson()
    b.combine({'FiredrakePoisson_np1': 'Firedrake',
               'DolfinPoisson_np1': 'DOLFIN'})
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = Poisson(name='DolfinPoissonParallel')
        b.combine_series([('np', np)], filename='DolfinPoisson')
        b.save()
        b = Poisson(name='FiredrakePoissonParallel')
        b.combine_series([('np', np)], filename='FiredrakePoisson')
        b.save()
        b = Poisson(name='PoissonParallel')
        b.combine({'FiredrakePoissonParallel': 'Firedrake',
                   'DolfinPoissonParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
