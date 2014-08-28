from pybench import Benchmark, parser

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06     # time step
theta = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Create a series of meshes that roughly double in number of DOFs
sizes = [125, 176, 250, 354, 500, 707, 1000]
cells = [2*x**2 for x in sizes]
regions = ['initial condition', 'Assemble cells', 'SNES solver execution']


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
    p = parser(description="Plot results for CahnHilliard benchmark")
    p.add_argument('-m', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    p.add_argument('-v', '--variant', nargs='+',
                   help='variants to plot')
    p.add_argument('-b', '--base', type=int,
                   help='index of size to use as base for parallel efficiency')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    groups = ['variant'] if len(variants) > 1 else []
    if args.sequential:
        b = CahnHilliard(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', variants), ('size', args.size or sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=groups,
               title='Cahn-Hilliard (single core, 2D)')
    if args.parallel:
        base = args.parallel.index(args.base or 1)
        efficiency = lambda xvals, yvals: [xvals[base]*yvals[base]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        for size in args.size or sizes:
            dofs = 2*(size+1)**2  # Factor 2 due to the mixed space

            def doflabel(n):
                if dofs > 1e6*n:
                    return '%.2fM' % (dofs/(1e6*n))
                elif dofs > 1e3*n:
                    return '%dk' % (dofs/(1e3*n))
                return str(dofs/n)
            title = 'Cahn-Hilliard (strong scaling, 2D, %.2fM DOFs)' % (dofs/1e6)
            xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.parallel]
            xlabel = 'Number of cores / DOFs per core'
            b = CahnHilliard(resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.parallel), ('variant', variants), ('size', [size])])
            b.plot(xaxis='np', regions=regions, figname='CahnHilliardStrong',
                   xlabel=xlabel, xticklabels=xticklabels, kinds='loglog',
                   groups=groups, title=title, trendline='perfect speedup')
            b.plot(xaxis='np', regions=regions, figname='CahnHilliardStrongEfficiency',
                   ylabel='Parallel efficiency w.r.t. %d cores' % args.parallel[base],
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot', hidexticks=range(2),
                   groups=groups, title=title, transform=efficiency, ymin=0)
            if 'DOLFIN' in variants:
                b.plot(xaxis='np', regions=regions, speedup=(1, 'DOLFIN'),
                       ylabel='Speedup relative to DOLFIN on 1 core',
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                       groups=groups, title=title)
