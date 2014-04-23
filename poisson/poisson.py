from pybench import Benchmark

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(5)]

class Poisson(Benchmark):

    params = [('dim', [dim]),
              ('degree', range(1, 4)),
              ('size', sizes)]
    meta = {'cells': [6*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'poisson'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    b = Poisson(name='DolfinPoissonParallel')
    b.combine_series([('np', [1, 2, 4, 6])], filename='DolfinPoisson')
    b.save()
    b = Poisson(name='FiredrakePoissonParallel')
    b.combine_series([('np', [1, 2, 4, 6])], filename='FiredrakePoisson')
    b.save()
    b = Poisson()
    b.combine({'FiredrakePoisson_np1': 'Firedrake', 'DolfinPoisson_np1': 'DOLFIN'})
    b.plot(xaxis='size')
    b = Poisson(name='PoissonParallel')
    b.combine({'FiredrakePoissonParallel': 'Firedrake', 'DolfinPoissonParallel': 'DOLFIN'})
    b.plot(xaxis='np')
