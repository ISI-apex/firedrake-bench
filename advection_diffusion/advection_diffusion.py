from pybench import Benchmark

dim = 2
# Create a series of meshes that roughly double in number of DOFs
sizes = [64, 88, 125, 176, 250]


class AdvectionDiffusion(Benchmark):

    params = [('dim', [dim]),
              ('degree', range(1, 4)),
              ('size', sizes)]
    meta = {'cells': [2*x**dim for x in sizes],
            'dofs': [(x+1)**dim for x in sizes]}
    method = 'advection_diffusion'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    regions = ['Firedrake advection RHS', 'Firedrake advection solve',
               'Firedrake diffusion RHS', 'Firedrake diffusion solve',
               'DOLFIN advection RHS', 'DOLFIN advection solve',
               'DOLFIN diffusion RHS', 'DOLFIN diffusion solve',
               'Firedrake timestepping', 'DOLFIN timestepping']
    b = AdvectionDiffusion(name='DolfinAdvectionDiffusionParallel')
    b.combine_series([('np', [1, 2, 3, 6])], filename='DolfinAdvectionDiffusion')
    b.save()
    b = AdvectionDiffusion(name='FiredrakeAdvectionDiffusionParallel')
    b.combine_series([('np', [1, 2, 3, 6])], filename='FiredrakeAdvectionDiffusion')
    b.save()
    b = AdvectionDiffusion()
    b.combine({'FiredrakeAdvectionDiffusion_np1': 'Firedrake',
               'DolfinAdvectionDiffusion_np1': 'DOLFIN'})
    b.plot(xaxis='size', regions=regions, xlabel='mesh size (cells)',
           xvalues=b.meta['cells'])
    b.plot(xaxis='degree', regions=regions, xlabel='Polynomial degree')
    b = AdvectionDiffusion(name='AdvectionDiffusionParallel')
    b.combine({'FiredrakeAdvectionDiffusionParallel': 'Firedrake',
               'DolfinAdvectionDiffusionParallel': 'DOLFIN'})
    b.plot(xaxis='np', regions=regions)
