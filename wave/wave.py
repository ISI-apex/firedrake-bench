from pybench import Benchmark, parser

scale = [1.0, 0.71, 0.5, 0.35, 0.25, 0.18, 0.125]
cells = dict(zip(scale, [42254, 82072, 169418, 337266, 679624, 1309528, 2716428]))
vertices = dict(zip(scale, [21119, 41028, 84701, 168625, 339804, 656622, 1358206]))
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
cells[0.29] = 522954
cells[0.2] = 1059980
cells[0.16] = 1617232
cells[0.14] = 2172644
cells[0.1] = 4247104
cells[0.07] = 8774894
cells[0.05] = 16508492
vertices[0.29] = 261469
vertices[0.2] = 529982
vertices[0.16] = 808608
vertices[0.14] = 1086314
vertices[0.1] = 2123544
vertices[0.07] = 4387439
vertices[0.05] = 8254238

if __name__ == '__main__':
    p = parser(description="Plot results for explicit wave benchmark")
    p.add_argument('-d', '--scale', type=float, nargs='+',
                   help='mesh scales to plot')
    args = p.parse_args()
    if args.sequential:
        b = Wave(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('scale', args.scale or scale),
                          ('variant', ['Firedrake', 'DOLFIN'])])
        b.plot(xaxis='scale', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (single core, 2D, mass lumping)')
    if args.weak:
        dofs = lambda n: vertices[round(0.5/n**.5, 2)]

        def doflabel(n):
            return '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
        b = Wave(benchmark='WaveWeak', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.weak), ('variant', ['Firedrake'])], filename='Wave')
        dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
        b.plot(xaxis='np', regions=regions,
               xlabel='Number of processors / DOFs (DOFs per processor: %dk)' % dpp,
               xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.weak],
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (weak scaling, 2D, mass lumping)')
    if args.parallel:
        b = Wave(benchmark='WaveParallel', resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.parallel), ('scale', args.scale or scale),
                          ('variant', ['Firedrake', 'DOLFIN'])], filename='Wave')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Explicit wave equation (strong scaling, 2D, mesh scaling %(scale)s)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Explicit wave equation (strong scaling, 2D, mesh scaling %(scale)s)')
