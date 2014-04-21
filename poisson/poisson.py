from pybench import Benchmark


class Poisson(Benchmark):

    params = {'degree': range(1, 4),
              'size': [2**x for x in range(4, 7)],
              'dim': [3]}
    method = 'poisson'
