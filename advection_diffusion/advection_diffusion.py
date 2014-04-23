from pybench import Benchmark

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [64, 125, 250, 500]


class AdvectionDiffusion(Benchmark):

    params = [('dim', [dim]),
              ('degree', range(1, 4)),
              ('size', sizes)]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'advection_diffusion'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
