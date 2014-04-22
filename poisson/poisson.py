from pybench import Benchmark


class Poisson(Benchmark):

    params = {'degree': range(1, 4),
              'size': [2**x for x in range(4, 7)],
              'dim': [3]}
    method = 'poisson'

if __name__ == '__main__':
    b = Poisson()
    b.combine({'FiredrakePoisson': 'Firedrake', 'DolfinPoisson': 'DOLFIN'})
    b.plot(xaxis='size')
