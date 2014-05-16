from pybench import Benchmark
from itertools import product

r0 = ['DOLFIN', 'Firedrake']
r1 = ['mass', 'laplace', 'helmholtz']
r2 = ['0', '1', '2', '3']

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(3)]


class Assembly(Benchmark):

    params = [('dim', [dim]),
              ('degree', [1, 2, 3]),
              ('size', sizes),
              ('fs', ['scalar', 'vector'])]
    meta = {'cells': [6*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'assembly'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = map(' '.join, product(r1, ['premult'], r2))

if __name__ == '__main__':
    import sys
    b = Assembly()
    b.combine({'FiredrakeAssembly_np1': 'Firedrake',
               'DolfinAssembly_np1': 'DOLFIN'})
    # Create separate plots for mass, laplace, helmholtz
    for r in r1:
        regions = map(' '.join, product(r0, [r], ['premult'], r2))
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=b.meta['cells'], figname=b.name + '_' + r, wscale=0.7)
        b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
               figname=b.name + '_' + r, wscale=0.7)
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = Assembly(name='DolfinAssemblyParallel')
        b.combine_series([('np', np)], filename='DolfinAssembly')
        b.save()
        b = Assembly(name='FiredrakeAssemblyParallel')
        b.combine_series([('np', np)], filename='FiredrakeAssembly')
        b.save()
        b = Assembly(name='AssemblyParallel')
        b.combine({'FiredrakeAssemblyParallel': 'Firedrake',
                   'DolfinAssemblyParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
