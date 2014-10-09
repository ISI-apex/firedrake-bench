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
    p.add_argument('-d', '--degree', type=int, nargs='+',
                   help='polynomial degrees to plot')
    p.add_argument('-m', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    p.add_argument('-v', '--variant', nargs='+', help='variants to plot')
    args = p.parse_args()
    variants = args.variant or ['Firedrake', 'DOLFIN']
    groups = ['variant'] if len(variants) > 1 else []
    degrees = args.degree or [1, 2, 3]
    if args.sequential:
        b = AdvectionDiffusion(resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', [1]), ('variant', variants),
                          ('size', args.size or sizes)])
        b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
               xvalues=cells, kinds='plot,loglog', groups=groups,
               title='Advection-diffusion (sequential, 2D, polynomial degree %(degree)d)')
        b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
               kinds='bar,barlog', groups=groups,
               title='Advection-diffusion (sequential, 2D, mesh size %(size)s**2)')
    if args.weak:
        for degree in degrees:
            dofs = lambda n: (int((1e4*n)**(1./dim))*degree+1)**dim

            def doflabel(n):
                return '%.1fM' % (dofs(n)/1e6) if dofs(n) > 1e6 else '%dk' % (dofs(n)/1e3)
            b = AdvectionDiffusion(benchmark='AdvectionDiffusionWeak',
                                   params=[('degree', [degree])],
                                   resultsdir=args.resultsdir, plotdir=args.plotdir)
            b.combine_series([('np', args.weak), ('variant', variants)],
                             filename='AdvectionDiffusion')
            dpp = dofs(args.weak[-1])/(1000*args.weak[-1])
            b.plot(xaxis='np', regions=regions,
                   xlabel='Number of processors / DOFs (DOFs per processor: %dk)' % dpp,
                   xticklabels=['%d\n%s' % (n, doflabel(n)) for n in args.weak],
                   kinds='plot,loglog', groups=groups,
                   title='Advection-diffusion (weak scaling, 2D, polynomial degree %(degree)d)')
    if args.parallel:
        b = AdvectionDiffusion(benchmark='AdvectionDiffusionParallel',
                               resultsdir=args.resultsdir, plotdir=args.plotdir)
        b.combine_series([('np', args.parallel), ('variant', variants),
                          ('size', args.size or sizes)], filename='AdvectionDiffusion')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=groups,
               title='Advection-diffusion (strong scaling, 2D, degree %(degree)d, mesh size %(size)s**2)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=groups, speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Advection-diffusion (strong scaling, 2D, degree %(degree)d, mesh size %(size)s**2)')
