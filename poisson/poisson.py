from pybench import Benchmark, parser

dim = 3
# Create a series of meshes that roughly double in number of DOFs
sizes = [int((1e4*2**x)**(1./dim)) + 1 for x in range(4)]
cells = [6*x**dim for x in sizes]
r0 = ['DOLFIN', 'Firedrake']
regions = ['matrix assembly', 'rhs assembly', 'solve']


class Poisson(Benchmark):

    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'matrix assembly': {'marker': '>'},
                 'rhs assembly': {'marker': '<'},
                 'solve': {'marker': 'D'}}
    method = 'poisson'
    benchmark = 'Poisson'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    args = parser(description="Plot results for Poisson benchmark").parse_args()
    if args.sequential:
        b = Poisson(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN']),
                          ('degree', [1, 2, 3]), ('size', sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=['variant'],
               title='Poisson (single core, %(dim)dD, polynomial degree %(degree)d)')
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot', groups=['variant', 'degree'],
               ylabel='Speedup relative to DOLFIN', speedup=('DOLFIN',),
               title='Poisson (single core, %(dim)dD)')
    if args.weak:
        b = Poisson(benchmark='PoissonWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('variant', ['Firedrake']),
                          ('degree', [1, 2, 3])], filename='Poisson')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Poisson (weak scaling, polynomial degree %(degree)d, 3D)')
    if args.parallel:
        b = Poisson(benchmark='PoissonParallel', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.parallel), ('variant', ['Firedrake', 'DOLFIN']),
                          ('degree', [1, 2, 3]), ('size', sizes)],
                         filename='Poisson')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Poisson (single node, %(dim)dD, polynomial degree %(degree)d, mesh size %(size)d**%(dim)d)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Poisson (single node, %(dim)dD, polynomial degree %(degree)d, mesh size %(size)d**%(dim)d)')
