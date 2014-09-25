from pybench import Benchmark, parser

scale = [1.0, 0.707, 0.5, 0.354, 0.25, 0.177, 0.125]
cells = dict(zip(scale, [42254, 91600, 169418, 335842, 679624, 1380102, 2716428]))
vertices = dict(zip(scale, [21119, 45792, 84701, 167913, 339804, 690043, 1358206]))
regions = ['p', 'phi']


class Wave(Benchmark):
    method = 'wave'
    benchmark = 'Wave'
    plotstyle = {'total': {'marker': '*'},
                 'mesh': {'marker': '+'},
                 'setup': {'marker': 'x'},
                 'p': {'marker': '>'},
                 'phi': {'marker': '<'},
                 'timestepping': {'marker': 'D'}}
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}
    profileregions = regions

# Weak scaling sizes
cells[0.577] = 133246
cells[0.408] = 266700
cells[0.316] = 446216
cells[0.289] = 510116
cells[0.204] = 997134
cells[0.158] = 1737978
cells[0.144] = 2000264
cells[0.102] = 4098480
cells[0.072] = 8216776
cells[0.051] = 16365888
cells[0.036] = 32832894
cells[0.026] = 62893912
vertices[0.577] = 66615
vertices[0.408] = 133342
vertices[0.316] = 223100
vertices[0.289] = 255050
vertices[0.204] = 498559
vertices[0.158] = 868981
vertices[0.144] = 1000124
vertices[0.102] = 2049232
vertices[0.072] = 4108380
vertices[0.051] = 8182936
vertices[0.036] = 16416447
vertices[0.026] = 31446948

if __name__ == '__main__':
    p = parser(description="Plot results for explicit wave benchmark")
    p.add_argument('-d', '--scale', type=float, nargs='+',
                   help='mesh scales to plot')
    p.add_argument('-v', '--variant', nargs='+',
                   help='variants to plot')
    p.add_argument('-b', '--base', type=int,
                   help='index of size to use as base for parallel efficiency')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    groups = ['variant'] if len(variants) > 1 else []
    if args.sequential:
        b = Wave(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('scale', args.scale or scale),
                          ('variant', variants)])
        b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=groups,
               title='Explicit wave equation (single core, 2D, mass lumping)')
    if args.weak:
        base = args.weak.index(args.base or 1)
        scale = args.scale[0] if args.scale else 1.0
        dofs = lambda n: vertices[round(scale/n**.5, 3)]
        doflabel = lambda n: '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
        b = Wave(benchmark='WaveWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('weak', [scale]), ('variant', variants)],
                         filename='Wave')
        dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
        xlabel = 'Number of cores / DOFs (DOFs per core: %dk)' % dpp
        xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.weak]
        title = 'Explicit wave equation (weak scaling, 2D, mass lumping)'
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
                   figname='WaveWeakEfficiency',
                   ylabel='Parallel efficiency w.r.t. %d cores' % args.weak[base],
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                   groups=groups, title=title, transform=efficiency, ymin=0)
    if args.parallel:
        base = args.parallel.index(args.base or 1)
        efficiency = lambda xvals, yvals: [xvals[base]*yvals[base]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        for sc in args.scale or scale:
            dofs = vertices[sc]

            def doflabel(n):
                if dofs > 1e6*n:
                    return '%.2fM' % (dofs/(1e6*n))
                elif dofs > 1e3*n:
                    return '%dk' % (dofs/(1e3*n))
                return str(dofs/n)
            title = 'Explicit wave equation (strong scaling, 2D, %.2fM cells, %.2fM DOFs)' % (cells[sc]/1e6, dofs/1e6)
            xticklabels = ['%d\n%s' % (n, doflabel(n)) for n in args.parallel]
            xlabel = 'Number of cores / DOFs per core'
            b = Wave(benchmark='WaveStrong', resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.parallel), ('scale', [sc]),
                              ('variant', variants)], filename='Wave')
            b.plot(xaxis='np', regions=regions, xlabel=xlabel,
                   xticklabels=xticklabels, kinds='loglog',
                   groups=groups, title=title, trendline='perfect speedup')
            b.plot(xaxis='np', regions=regions, figname='WaveStrongEfficiency',
                   ylabel='Parallel efficiency w.r.t. %d cores' % args.parallel[base],
                   xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                   groups=groups, title=title, transform=efficiency, ymin=0)
            if 'DOLFIN' in variants:
                b.plot(xaxis='np', regions=regions, speedup=(1, 'DOLFIN'),
                       ylabel='Speedup relative to DOLFIN on 1 core',
                       xlabel=xlabel, xticklabels=xticklabels, kinds='plot',
                       groups=groups, title=title)
