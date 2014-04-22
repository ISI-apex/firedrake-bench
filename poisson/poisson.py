from pybench import Benchmark


class Poisson(Benchmark):

    params = {'degree': range(1, 4),
              'size': [2**x for x in range(4, 7)],
              'dim': [3]}
    method = 'poisson'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    b = Poisson()
    b.combine({'FiredrakePoisson': 'Firedrake', 'DolfinPoisson': 'DOLFIN'})
    b.plot(xaxis='size')
