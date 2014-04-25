from pybench import Benchmark


class NavierStokes(Benchmark):

    params = [('scale', [0.8, 0.56, 0.4, 0.28, 0.2]),
              ('preassemble', [True, False])]
    meta = {'cells': [30906, 63432, 124390, 253050, 496156, 1013484, 1981076],
            'dofs': [15451, 31714, 62193, 126523, 248076, 506740, 990536]}
    method = 'navier_stokes'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
