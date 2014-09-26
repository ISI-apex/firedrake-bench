from pybench import Benchmark, parser
from itertools import product

scale = [1.0, 0.707, 0.5, 0.354, 0.25, 0.177, 0.125]
cells = dict(zip(scale, [19810, 39736, 79034, 158960, 318030, 631774, 1271436]))
vertices = dict(zip(scale, [9906, 19869, 39518, 79481, 159016, 315888, 635719]))
r1 = ['tentative velocity', 'pressure correction', 'velocity correction']
r2 = ['RHS', 'solve']

# Weak scaling sizes
cells[4.0] = 1240
cells[2.828] = 2444
cells[2.0] = 5046
cells[1.414] = 9960
cells[0.577] = 59614
cells[0.408] = 119812
cells[0.316] = 198694
cells[0.289] = 238048
cells[0.204] = 478240
cells[0.158] = 793468
cells[0.144] = 957686
cells[0.102] = 1905144
cells[0.072] = 3828300
cells[0.051] = 7621454
cells[0.036] = 15302786
cells[0.026] = 29377542
vertices[4.0] = 618
vertices[2.828] = 1220
vertices[2.0] = 2521
vertices[1.414] = 4978
vertices[0.577] = 29805
vertices[0.408] = 59904
vertices[0.316] = 99345
vertices[0.289] = 119022
vertices[0.204] = 239118
vertices[0.158] = 396732
vertices[0.144] = 478841
vertices[0.102] = 952570
vertices[0.072] = 1914148
vertices[0.051] = 3810725
vertices[0.036] = 7651391
vertices[0.026] = 14688769


class NavierStokes(Benchmark):

    method = 'navier_stokes'
    benchmark = 'NavierStokes'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'matrix assembly': {'marker': 'p'},
                 'timestepping': {'marker': 'o'},
                 'tentative velocity RHS': {'marker': '^'},
                 'tentative velocity solve': {'marker': 'v'},
                 'pressure correction RHS': {'marker': 's'},
                 'pressure correction solve': {'marker': 'D'},
                 'velocity correction RHS': {'marker': '>'},
                 'velocity correction solve': {'marker': '<'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = ['matrix assembly'] + map(' '.join, product(r1, r2))

if __name__ == '__main__':
    p = parser(description="Plot results for Navier-Stokes benchmark")
    p.add_argument('-m', '--scale', type=float, nargs='+',
                   help='mesh scales to plot')
    p.add_argument('-v', '--variant', nargs='+', help='variants to plot')
    p.add_argument('-b', '--base', type=int,
                   help='index of size to use as base for parallel efficiency')
    p.add_argument('-r', '--region', nargs='+', help='regions to plot')
    args = p.parse_args()
    regions = args.region or map(' '.join, product(r1, r2))
    variants = args.variant or ['Firedrake', 'DOLFIN']
    groups = ['variant'] if len(variants) > 1 else []
    scales = args.scale or scale
    if args.sequential:
        b = NavierStokes(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('scale', scales), ('variant', variants)])
        b.plot(xaxis='scale', regions=regions, xlabel='DOFs', groups=groups,
               xvalues=[vertices[s] for s in scales], kinds='plot,loglog',
               title='Navier-Stokes (sequential, 2D, P2-P1)')
    if args.weak:
        base = args.weak.index(args.base or 1)
        scale = args.scale[0] if args.scale else 1.0
        dofs = lambda n: vertices[round(scale/n**.5, 3)]
        doflabel = lambda n: '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
        b = NavierStokes(benchmark='NavierStokesWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('weak', [scale]), ('variant', variants)],
                         filename='NavierStokes')
        dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
        xlabel = 'Number of cores / DOFs (DOFs per core: %dk)' % dpp
        xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.weak]
        title = 'Navier-Stokes (weak scaling, 2D, P2-P1)'
        if args.base:
            efficiency = lambda xvals, yvals: [yvals[0]/y for x, y in zip(xvals, yvals)]
            subplotargs = {(0, 0): {'xvals': args.weak[:base+1],
                                    'hideyticks': range(2),
                                    'title': 'intra node: 1-%d cores' % args.base,
                                    'xticklabels': xticklabels[:base+1],
                                    'xlabel': None},
                           (0, 1): {'xvals': args.weak[base:],
                                    'title': 'inter node: %d-%d cores' % (args.base, args.weak[-1]),
                                    'xticklabels': xticklabels[base:],
                                    'xlabel': None,
                                    'ylabel': None},
                           (1, 0): {'xvals': args.weak[:base+1],
                                    'xticklabels': xticklabels[:base+1],
                                    'transform': efficiency,
                                    'ymin': 0,
                                    'xlabel': 'Number of cores / DOFs',
                                    'ylabel': 'Parallel efficiency w.r.t. 1/%d cores' % args.weak[base]},
                           (1, 1): {'xvals': args.weak[base:],
                                    'xticklabels': xticklabels[base:],
                                    'transform': efficiency,
                                    'hidexticks': range(2),
                                    'ymin': 0,
                                    'xlabel': 'DOFs per core: %dk' % dpp,
                                    'ylabel': None}}
            b.plot(xaxis='np', regions=regions, kinds='plot', groups=groups,
                   title=title, subplots=(2, 2), sharex='col', sharey='row',
                   hspace=0.02, wspace=0.02, subplotargs=subplotargs)
        else:
            efficiency = lambda xvals, yvals: [yvals[base]/y for x, y in zip(xvals, yvals)]
            b.plot(xaxis='np', regions=regions, hidexticks=range(5),
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot,loglog',
                   groups=groups, title=title)
            b.plot(xaxis='np', regions=regions, hidexticks=range(5),
                   figname='NavierStokesWeakEfficiency',
                   ylabel='Parallel efficiency w.r.t. %d cores' % args.weak[base],
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                   groups=groups, title=title, transform=efficiency, ymin=0)
    if args.parallel:
        base = args.parallel.index(args.base or 1)
        efficiency = lambda xvals, yvals: [xvals[base]*yvals[base]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        for sc in scales:
            dofs = vertices[sc]

            def doflabel(n):
                if dofs > 1e6*n:
                    return '%.2fM' % (dofs/(1e6*n))
                elif dofs > 1e3*n:
                    return '%dk' % (dofs/(1e3*n))
                return str(dofs/n)
            title = 'Navier-Stokes (strong scaling, 2D, %.2fM cells, %.2fM DOFs)' % (cells[sc]/1e6, dofs/1e6)
            xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.parallel]
            xlabel = 'Number of cores / DOFs per core'
            b = NavierStokes(benchmark='NavierStokesStrong', resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.parallel), ('scale', [sc]),
                              ('variant', variants)], filename='NavierStokes')
            b.plot(xaxis='np', regions=regions, xlabel=xlabel,
                   xticklabels=xticklabels, kinds='loglog',
                   groups=groups, title=title, trendline='perfect speedup')
            b.plot(xaxis='np', regions=regions, figname='NavierStokesStrongEfficiency',
                   ylabel='Parallel efficiency w.r.t. %d cores' % args.parallel[base],
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                   groups=groups, title=title, transform=efficiency, ymin=0)
            if 'DOLFIN' in variants:
                b.plot(xaxis='np', regions=regions, speedup=(1, 'DOLFIN'),
                       ylabel='Speedup relative to DOLFIN on 1 core',
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                       groups=groups, title=title)
