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
    p = parser(description="Plot results for Poisson benchmark")
    p.add_argument('-d', '--degree', type=int, nargs='+',
                   help='polynomial degrees to plot')
    p.add_argument('-m', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    p.add_argument('-v', '--variant', nargs='+',
                   help='variants to plot')
    p.add_argument('-b', '--base', type=int,
                   help='index of size to use as base for parallel efficiency')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    groups = ['variant'] if len(variants) > 1 else []
    degrees = args.degree or [1, 2, 3]
    if args.sequential:
        b = Poisson(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', variants),
                          ('degree', degrees), ('size', args.size or sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=groups,
               title='Poisson (sequential, 3D, polynomial degree %(degree)d)')
        if 'DOLFIN' in variants:
            b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
                   xvalues=cells, kinds='plot', groups=groups,
                   ylabel='Speedup relative to DOLFIN', speedup=('DOLFIN',),
                   title='Poisson (sequential, 3D)')
    if args.weak:
        base = args.weak.index(args.base or 1)
        size = args.size[0] if args.size else 10000
        for degree in degrees:
            dofs = lambda n: (int((size*n)**(1./dim))*degree+1)**dim
            doflabel = lambda n: '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
            b = Poisson(benchmark='PoissonWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.weak), ('weak', [size]),
                              ('variant', variants), ('degree', [degree])],
                             filename='Poisson')
            dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
            xlabel = 'Number of cores / DOFs (DOFs per core: %dk)' % dpp
            xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.weak]
            title = 'Poisson (weak scaling, polynomial degree %d, 3D)' % degree
            if args.base:
                efficiency = lambda xvals, yvals: [yvals[0]/y for x, y in zip(xvals, yvals)]
                subplotargs = {(0, 0): {'xvals': args.weak[:base+1],
                                        'title': 'intra node: 1-%d cores' % args.base,
                                        'xticklabels': xticklabels[:base+1],
                                        'xlabel': None,
                                        'legend': False},
                               (0, 1): {'xvals': args.weak[base:],
                                        'title': 'inter node: %d-%d cores' % (args.base, args.weak[-1]),
                                        'xticklabels': xticklabels[base:],
                                        'xlabel': None,
                                        'ylabel': None,
                                        'legend': False},
                               (1, 0): {'xvals': args.weak[:base+1],
                                        'xticklabels': xticklabels[:base+1],
                                        'transform': efficiency,
                                        'ymin': 0.0,
                                        'xlabel': 'Number of cores / DOFs',
                                        'ylabel': 'Parallel efficiency w.r.t. 1/%d cores' % args.weak[base],
                                        'legend': False},
                               (1, 1): {'xvals': args.weak[base:],
                                        'xticklabels': xticklabels[base:],
                                        'transform': efficiency,
                                        'hidexticks': range(3),
                                        'ymin': 0.0,
                                        'xlabel': 'DOFs per core: %dk' % dpp,
                                        'ylabel': None,
                                        'legend': False}}
                b.plot(xaxis='np', regions=regions, kinds='plot', groups=groups,
                       title=title, subplots=(2, 2), sharex='col', sharey='row',
                       hspace=0.02, wspace=0.02, subplotargs=subplotargs,
                       legend={'loc': 'upper left', 'bbox_to_anchor': (0.1, 0.9)})
            else:
                efficiency = lambda xvals, yvals: [yvals[base]/y for x, y in zip(xvals, yvals)]
                b.plot(xaxis='np', regions=regions, hidexticks=range(6),
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot,loglog',
                       groups=groups, title=title)
                b.plot(xaxis='np', regions=regions, hidexticks=range(6),
                       figname='PoissonWeakEfficiency',
                       ylabel='Parallel efficiency w.r.t. %d cores' % args.weak[base],
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                       groups=groups, title=title, transform=efficiency, ymin=0)
    if args.parallel:
        base = args.parallel.index(args.base or 1)
        efficiency = lambda xvals, yvals: [xvals[base]*yvals[base]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        for degree in degrees:
            for size in args.size or sizes:
                dofs = (size*degree+1)**dim

                def doflabel(n):
                    if dofs > 1e6*n:
                        return '%.2fM' % (dofs/(1e6*n))
                    elif dofs > 1e3*n:
                        return '%dk' % (dofs/(1e3*n))
                    return str(dofs/n)
                title = 'Poisson (strong scaling, 3D, polynomial degree %d, %.2fM DOFs)' % (degree, dofs/1e6)
                xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.parallel]
                xlabel = 'Number of cores / DOFs per core'
                b = Poisson(benchmark='PoissonStrong', resultsdir=args.resultsdir, plotdir=args.plotdir)
                b.combine_series([('np', args.parallel), ('variant', variants),
                                  ('degree', [degree]), ('size', [size])],
                                 filename='Poisson')
                b.plot(xaxis='np', regions=regions, xlabel=xlabel,
                       xticklabels=xticklabels, kinds='loglog',
                       groups=groups, title=title, trendline='perfect speedup')
                b.plot(xaxis='np', regions=regions, figname='PoissonStrongEfficiency',
                       ylabel='Parallel efficiency w.r.t. %d cores' % args.parallel[base],
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot', hidexticks=range(2),
                       groups=groups, title=title, transform=efficiency, ymin=0)
                if 'DOLFIN' in variants:
                    b.plot(xaxis='np', regions=regions, speedup=(1, 'DOLFIN'),
                           ylabel='Speedup relative to DOLFIN on 1 core',
                           xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                           groups=groups, title=title)
