from pybench import Benchmark

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [64, 88, 125, 176, 250]
profileregions = ['mesh', 'setup', 'timestepping']


class CahnHilliard(Benchmark):

    params = [('degree', [1, 2]),
              ('size', sizes),
              ('steps', [1])]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'cahn_hilliard'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = profileregions

if __name__ == '__main__':
    import sys
    from itertools import product
    regions = map(' '.join, product(['DOLFIN', 'Firedrake'], profileregions))
    b = CahnHilliard()
    b.combine({'FiredrakeCahnHilliard_np1': 'Firedrake',
               'DolfinCahnHilliard_np1': 'DOLFIN'})
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = CahnHilliard(name='DolfinCahnHilliardParallel')
        b.combine_series([('np', np)], filename='DolfinCahnHilliard')
        b.save()
        b = CahnHilliard(name='FiredrakeCahnHilliardParallel')
        b.combine_series([('np', np)], filename='FiredrakeCahnHilliard')
        b.save()
        b = CahnHilliard(name='CahnHilliardParallel')
        b.combine({'FiredrakeCahnHilliardParallel': 'Firedrake',
                   'DolfinCahnHilliardParallel': 'DOLFIN'})
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog')
