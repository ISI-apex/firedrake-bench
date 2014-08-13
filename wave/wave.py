from pybench import Benchmark, parser

scale = [1.0, 0.707, 0.5, 0.354, 0.25, 0.177, 0.125]
cells = dict(zip(scale, [42254, 91600, 169418, 335842, 679624, 1380102, 2716428]))
vertices = dict(zip(scale, [21119, 45792, 84701, 167913, 339804, 690043, 1358206]))
regions = ['setup', 'timestepping', 'p', 'phi']


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

if __name__ == '__main__':
    p = parser(description="Plot results for explicit wave benchmark")
    p.add_argument('-d', '--scale', type=float, nargs='+',
                   help='mesh scales to plot')
    p.add_argument('-v', '--variant', nargs='+',
                   help='variants to plot')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    if args.sequential:
        b = Wave(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('scale', args.scale or scale),
                          ('variant', variants)])
        b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (single core, 2D, mass lumping)')
    if args.weak:
        dofs = lambda n: vertices[round(1./n**.5, 3)]

        def doflabel(n):
            return '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
        b = Wave(benchmark='WaveWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('variant', variants)], filename='Wave')
        dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
        b.plot(xaxis='np', regions=regions,
               xlabel='Number of processors / DOFs (DOFs per processor: %dk)' % dpp,
               xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.weak],
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (weak scaling, 2D, mass lumping)')
    if args.parallel:
        b = Wave(benchmark='WaveParallel', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.parallel), ('scale', args.scale or scale),
                          ('variant', variants)], filename='Wave')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (strong scaling, 2D, mesh scaling %(scale)s)')
        if 'DOLFIN' in variants:
            b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
                   kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
                   ylabel='Speedup relative to DOLFIN on 1 core',
                   title='Explicit wave equation (strong scaling, 2D, mesh scaling %(scale)s)')
