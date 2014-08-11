from pybench import Benchmark, parser

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [125, 176, 250, 354, 500]
cells = [2*x**dim for x in sizes]
dofs = [(x+1)**dim for x in sizes]
regions = ['advection RHS', 'diffusion RHS', 'advection solve', 'diffusion solve']


class AdvectionDiffusion(Benchmark):

    params = [('degree', range(1, 4))]
    method = 'advection_diffusion'
    benchmark = 'AdvectionDiffusion'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'advection matrix': {'marker': '>'},
                 'diffusion matrix': {'marker': '<'},
                 'timestepping': {'marker': 'o'},
                 'advection RHS': {'marker': '^'},
                 'diffusion RHS': {'marker': 'v'},
                 'advection solve': {'marker': 's'},
                 'diffusion solve': {'marker': 'D'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

if __name__ == '__main__':
    p = parser(description="Plot results for advection-diffusion benchmark")
    p.add_argument('-d', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    args = p.parse_args()
    if args.sequential:
        b = AdvectionDiffusion(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN']),
                          ('size', args.size or sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=['variant'],
               title='Advection-diffusion (sequential, 2D, polynomial degree %(degree)d)')
        b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
               kinds='bar,barlog', groups=['variant'],
               title='Advection-diffusion (sequential, 2D, mesh size %(size)s**2)')
    if args.weak:
        b = AdvectionDiffusion(benchmark='AdvectionDiffusionWeak',
                               resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('variant', ['Firedrake'])],
                         filename='AdvectionDiffusion')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Advection-diffusion (weak scaling, 2D, polynomial degree %(degree)d)')
    if args.parallel:
        b = AdvectionDiffusion(benchmark='AdvectionDiffusionParallel',
                               resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.parallel), ('variant', ['Firedrake', 'DOLFIN']),
                          ('size', args.size or sizes)], filename='AdvectionDiffusion')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Advection-diffusion (strong scaling, 2D, degree %(degree)d, mesh size %(size)s**2)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Advection-diffusion (strong scaling, 2D, degree %(degree)d, mesh size %(size)s**2)')
