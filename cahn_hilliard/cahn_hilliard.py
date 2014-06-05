from pybench import Benchmark

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [125, 176, 250, 354, 500, 707, 1000]
regions = ['mesh', 'setup', 'timestepping']


class CahnHilliard(Benchmark):

    params = [('degree', [1]),
              ('size', sizes),
              ('steps', [1])]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'cahn_hilliard'
    name = 'CahnHilliard'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': 's'},
                 'setup': {'marker': 'D'},
                 'timestepping': {'marker': 'o'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    import sys
    b = CahnHilliard()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN'])])
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'])
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = CahnHilliard(name='CahnHilliardParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN'])],
                         filename='CahnHilliard')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'])
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core')
