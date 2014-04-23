from pybench import Benchmark

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(5)]

class Poisson(Benchmark):

    params = [('dim', [dim]),
              ('degree', range(1, 4)),
              ('size', sizes)]
    method = 'poisson'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    Poisson().main(combine={'FiredrakePoisson': 'Firedrake',
                            'DolfinPoisson': 'DOLFIN'},
                   plot=['size'])
