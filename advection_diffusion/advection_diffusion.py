from pybench import Benchmark


class AdvectionDiffusion(Benchmark):

    params = [('degree', range(1, 4)),
              ('scale', [1.0, 0.71, 0.5, 0.35, 0.25])]
    meta = {'cells': [26386, 52166, 105418, 216162, 422660],
            'dofs': [13192, 26082, 52708, 108080, 211327]}
    method = 'advection_diffusion'
    profilegraph = {'format': 'svg,pdf',
                    'node_threshold': 2.0}

if __name__ == '__main__':
    regions = ['Firedrake advection RHS', 'Firedrake advection solve',
               'Firedrake diffusion RHS', 'Firedrake diffusion solve',
               'DOLFIN advection RHS', 'DOLFIN advection solve',
               'DOLFIN diffusion RHS', 'DOLFIN diffusion solve']
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
