from pybench import Benchmark

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [125, 176, 250, 354, 500]
regions = ['advection RHS', 'diffusion RHS', 'advection solve', 'diffusion solve']


class AdvectionDiffusion(Benchmark):

    params = [('degree', range(1, 4))]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
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
    import sys
    b = AdvectionDiffusion()
    b.combine_series([('np', [1]), ('variant', ['Firedrake', 'DOLFIN']), ('size', sizes)])
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'], kinds='plot,loglog', groups=['variant'],
           title='Advection-diffusion (single core, 2D, polynomial degree %(degree)d)')
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree',
           kinds='bar,barlog', groups=['variant'],
           title='Advection-diffusion (single core, 2D, mesh size %(size)s**2)')
    if len(sys.argv) > 1:
        np = map(int, sys.argv[1:])
        b = AdvectionDiffusion(benchmark='AdvectionDiffusionParallel')
        b.combine_series([('np', np), ('variant', ['Firedrake', 'DOLFIN']), ('size', sizes)],
                         filename='AdvectionDiffusion')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot,loglog', groups=['variant'],
               title='Advection-diffusion (single node, 2D, degree %(degree)d, mesh size %(size)s**2)')
        b.plot(xaxis='np', regions=regions, xlabel='Number of processors',
               kinds='plot', groups=['variant'], speedup=(1, 'DOLFIN'),
               ylabel='Speedup relative to DOLFIN on 1 core',
               title='Advection-diffusion (single node, 2D, degree %(degree)d, mesh size %(size)s**2)')
