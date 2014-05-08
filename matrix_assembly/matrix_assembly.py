from pybench import Benchmark

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(4)]
np = [1, 2, 3]


class MatrixAssembly(Benchmark):

    params = [('dim', [dim]),
              ('degree', [1, 2, 3]),
              ('size', sizes),
              ('fs', ['scalar', 'vector'])]
    meta = {'cells': [6*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'matrix_assembly'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    from itertools import product
    r0 = ['DOLFIN', 'Firedrake']
    r1 = ['assembly', 'reassembly', 'assembly bcs', 'reassembly bcs']
    regions = map(' '.join, product(r0, r1))
    b = MatrixAssembly(name='DolfinMatrixAssemblyParallel')
    b.combine_series([('np', np)], filename='DolfinMatrixAssembly')
    b.save()
    b = MatrixAssembly(name='FiredrakeMatrixAssemblyParallel')
    b.combine_series([('np', np)], filename='FiredrakeMatrixAssembly')
    b.save()
    b = MatrixAssembly()
    b.combine({'FiredrakeMatrixAssembly_np1': 'Firedrake',
               'DolfinMatrixAssembly_np1': 'DOLFIN'})
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    b = MatrixAssembly(name='MatrixAssemblyParallel')
    b.combine({'FiredrakeMatrixAssemblyParallel': 'Firedrake',
               'DolfinMatrixAssemblyParallel': 'DOLFIN'})
    b.plot(xaxis='np', regions=regions)