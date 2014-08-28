from pybench import Benchmark

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Create a series of meshes that roughly double in number of DOFs
sizes = [125, 176, 250, 354, 500, 707, 1000]
cells = [2*x**2 for x in sizes]
regions = ['mesh', 'initial condition', 'Assemble cells', 'SNES solver execution']


class CahnHilliard(Benchmark):

    method = 'cahn_hilliard'
    benchmark = 'CahnHilliard'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': 's'},
                 'initial condition': {'marker': 'D'},
                 'Assemble cells': {'marker': '^'},
                 'SNES solver execution': {'marker': 'o'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    import sys
    b = CahnHilliard()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN']), ('size', sizes)])
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=cells, kinds='plot,loglog', groups=['variant'],
           title='Cahn-Hilliard (single core, 2D, polynomial degree %(degree)d)')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = CahnHilliard(name='CahnHilliardParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN'], ('size', sizes))],
                         filename='CahnHilliard')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Cahn-Hilliard (single node, 2D, degree %(degree)d, mesh size %(size)d**2)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Cahn-Hilliard (single node, 2D, degree %(degree)d, mesh size %(size)d**2)')
