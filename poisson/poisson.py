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
    p.add_argument('-d', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    p.add_argument('-v', '--variant', nargs='+',
                   help='variants to plot')
    p.add_argument('-b', '--base', type=int,
                   help='index of size to use as base for parallel efficiency')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    if args.sequential:
        b = Poisson(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', variants),
                          ('degree', [1, 2, 3]), ('size', args.size or sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=['variant'],
               title='Poisson (sequential, 3D, polynomial degree %(degree)d)')
        if 'DOLFIN' in variants:
            b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
                   xvalues=cells, kinds='plot', groups=['variant', 'degree'],
                   ylabel='Speedup relative to DOLFIN', speedup=('DOLFIN',),
                   title='Poisson (sequential, 3D)')
    if args.weak:
        for degree in [1, 2, 3]:
            dofs = lambda n: (int((1e4*n)**(1./dim))*degree+1)**dim
            doflabel = lambda n: '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
            b = Poisson(benchmark='PoissonWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.weak), ('variant', variants),
                              ('degree', [degree])], filename='Poisson')
            dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
            b.plot(xaxis='np', regions=regions,
                   xlabel='Number of processors / DOFs (DOFs per processor: %dk)' % dpp,
                   xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.weak],
                   kinds='plot,loglog', groups=['variant'],
                   title='Poisson (weak scaling, polynomial degree %d, 3D)' % degree)
    if args.parallel:
        base = args.base or 0
        efficiency = lambda xvals, yvals: [xvals[base]*yvals[base]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        for degree in [1, 2, 3]:
            for size in args.size or sizes:
                dofs = (size*degree+1)**dim

                def doflabel(n):
                    if dofs > 1e6*n:
                        return '%.2fM' % (dofs/(1e6*n))
                    elif dofs > 1e3*n:
                        return '%dk' % (dofs/(1e3*n))
                    return str(dofs/n)
                b = Poisson(benchmark='PoissonParallel', resultsdir=args.resultsdir, plotdir=args.plotdir)
                b.combine_series([('np', args.parallel), ('variant', variants),
                                  ('degree', [degree]), ('size', [size])],
                                 filename='Poisson')
                b.plot(xaxis='np', regions=regions,
                       xlabel='Number of processors / DOFs per processor',
                       xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.parallel],
                       kinds='plot,loglog', groups=['variant'],
                       title='Poisson (strong scaling, 3D, polynomial degree %d, %.2fM DOFs)' % (degree, dofs/1e6))
                b.plot(xaxis='np', regions=regions, figname='PoissonEfficiency',
                       xlabel='Number of processors / DOFs per processor',
                       ylabel='Parallel efficiency w.r.t. %d cores' % args.parallel[base],
                       xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.parallel],
                       kinds='semilogx', groups=['variant'], transform=efficiency,
                       title='Poisson (strong scaling, 3D, polynomial degree %d, %.2fM DOFs)' % (degree, dofs/1e6))
                if 'DOLFIN' in variants:
                    b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
                           kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
                           ylabel='Speedup relative to DOLFIN on 1 core',
                           title='Poisson (strong scaling, 3D, polynomial degree %d, %.2fM DOFs)' % (degree, dofs/1e6))
