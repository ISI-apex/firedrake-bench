from pybench import Benchmark

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [64, 88, 125, 176, 250]


class CahnHilliard(Benchmark):

    params = [('degree', [1]),
              ('size', sizes),
              ('steps', [10])]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'cahn_hilliard'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
