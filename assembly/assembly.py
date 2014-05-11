from pybench import Benchmark

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(3)]
np = [1, 2, 3]


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

if __name__ == '__main__':
    from itertools import product
    r0 = ['DOLFIN', 'Firedrake']
    r1 = ['mass', 'laplace', 'helmholtz']
    r2 = ['assembly', 'assembly premult 1', 'assembly premult 2', 'assembly premult 3']
    regions = map(' '.join, product(r0, r1, r2))
    b = Assembly(name='DolfinAssemblyParallel')
    b.combine_series([('np', np)], filename='DolfinAssembly')
    b.save()
    b = Assembly(name='FiredrakeAssemblyParallel')
    b.combine_series([('np', np)], filename='FiredrakeAssembly')
    b.save()
    b = Assembly()
    b.combine({'FiredrakeAssembly_np1': 'Firedrake',
               'DolfinAssembly_np1': 'DOLFIN'})
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    b = Assembly(name='AssemblyParallel')
    b.combine({'FiredrakeAssemblyParallel': 'Firedrake',
               'DolfinAssemblyParallel': 'DOLFIN'})
    b.plot(xaxis='np', regions=regions)
